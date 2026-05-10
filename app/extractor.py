"""
extractor.py
------------
TITAN / CONCH feature extractor.

Responsibilities
----------------
- Download model weights from Hugging Face on first run (idempotent).
- Load the model once and reuse it across all batches.
- Accept batches of PIL images / numpy arrays.
- Return float32 feature vectors.
- Detect GPU automatically; fall back to CPU when unavailable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from app.config import (
    CONCH_HF_REPO,
    MODELS_CACHE_DIR,
    MODELS_CONCH_DIR,
    MODELS_TITAN_DIR,
    TITAN_HF_REPO,
)

logger = logging.getLogger(__name__)

# Set HF cache directory so weights land in the persistent volume
os.environ.setdefault("HF_HOME", str(MODELS_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODELS_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(MODELS_CACHE_DIR / "hub"))
# Force offline mode — all weights are local, no HuggingFace network calls needed
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


# ─── Device selection ─────────────────────────────────────────────────────────

def _select_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(
            "GPU detected: %s (VRAM %.1f GB)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    else:
        device = torch.device("cpu")
        logger.warning(
            "No GPU detected – running on CPU. "
            "This will be significantly slower. Intended for testing only."
        )
    return device


# ─── Model download ───────────────────────────────────────────────────────────

def _weights_present(model_dir: Path, required_files: list[str]) -> bool:
    return all((model_dir / f).exists() for f in required_files)


def download_titan_weights() -> None:
    """
    Download TITAN weights from Hugging Face into MODELS_TITAN_DIR.
    Skips the download if weights are already present.
    """
    from huggingface_hub import snapshot_download

    sentinel = MODELS_TITAN_DIR / "config.json"
    if sentinel.exists():
        logger.info("TITAN weights already present at %s – skipping download.", MODELS_TITAN_DIR)
        return

    logger.info("Downloading TITAN weights from %s …", TITAN_HF_REPO)
    MODELS_TITAN_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=TITAN_HF_REPO,
        local_dir=str(MODELS_TITAN_DIR),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.ot", "*.msgpack"],
    )
    logger.info("TITAN weights downloaded → %s", MODELS_TITAN_DIR)


def download_conch_weights() -> None:
    """
    CONCH v1.5 is bundled inside the TITAN repo as conch_v1_5_pytorch_model.bin.
    No separate download is needed — TITAN's return_conch() loads it at runtime.
    This function is kept for API compatibility but is effectively a no-op.
    """
    conch_bin = MODELS_TITAN_DIR / "conch_v1_5_pytorch_model.bin"
    if conch_bin.exists():
        logger.info("CONCH bundled inside TITAN at %s – no separate download needed.", conch_bin)
    else:
        logger.warning(
            "conch_v1_5_pytorch_model.bin not found in %s. "
            "Re-download TITAN weights via download_models.sh.",
            MODELS_TITAN_DIR,
        )


# ─── Transform ────────────────────────────────────────────────────────────────

def _build_transform(patch_size_px: int) -> transforms.Compose:
    """
    Standard image transform used by CONCH / TITAN.
    Resize → CenterCrop → ToTensor → Normalize (ImageNet stats).
    """
    return transforms.Compose([
        transforms.Resize(patch_size_px, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(patch_size_px),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


# ─── TITAN extractor ──────────────────────────────────────────────────────────

class TITANExtractor:
    """
    Wraps TITAN (built on CONCH visual encoder) for patch-level feature extraction.

    Usage
    -----
    extractor = TITANExtractor(patch_size_px=256)
    extractor.load()
    features = extractor.extract_batch(images)   # list[PIL.Image] → np.ndarray
    """

    MODEL_VERSION = "TITAN-v1"

    def __init__(self, patch_size_px: int = 256, batch_size: int = 32) -> None:
        self._patch_size_px = patch_size_px
        self._batch_size = batch_size
        self._device: Optional[torch.device] = None
        self._model: Optional[nn.Module] = None
        self._transform: Optional[transforms.Compose] = None
        self._feature_dim: Optional[int] = None

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            raise RuntimeError("Model not loaded – call load() first.")
        return self._feature_dim

    @property
    def model_version(self) -> str:
        return self.MODEL_VERSION

    def load(self) -> None:
        """
        Load TITAN from local weights directory.
        Weights must already be present (call download_titan_weights() first).
        """
        if self._model is not None:
            logger.debug("TITAN already loaded – reusing existing instance.")
            return

        self._device = _select_device()
        self._transform = _build_transform(self._patch_size_px)

        logger.info("Loading TITAN from %s …", MODELS_TITAN_DIR)
        try:
            import shutil
            from transformers import AutoModel

            # transformers copies only auto_map-registered .py files to its modules
            # cache, but conch_tokenizer.py (imported by text_transformer.py) is not
            # listed. Copy ALL .py files from the local weights dir to the cache so
            # every relative import resolves correctly.
            _hf_modules = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / "titan"
            _hf_modules.mkdir(parents=True, exist_ok=True)
            for _py in MODELS_TITAN_DIR.glob("*.py"):
                shutil.copy2(_py, _hf_modules / _py.name)

            # Fix conch_tokenizer.py: replace hardcoded HuggingFace repo name
            # with the local path so it works offline (no internet required).
            _tokenizer_cache = _hf_modules / "conch_tokenizer.py"
            if _tokenizer_cache.exists():
                _content = _tokenizer_cache.read_text()
                _fixed = _content.replace(
                    'from_pretrained("MahmoodLab/TITAN")',
                    f'from_pretrained("{MODELS_TITAN_DIR}", local_files_only=True)',
                ).replace(
                    f'from_pretrained("{MODELS_TITAN_DIR}")',
                    f'from_pretrained("{MODELS_TITAN_DIR}", local_files_only=True)',
                )
                if _fixed != _content:
                    _tokenizer_cache.write_text(_fixed)
                    logger.info("Patched conch_tokenizer.py to use local path: %s", MODELS_TITAN_DIR)

            logger.info("Copied %d .py files → %s", len(list(MODELS_TITAN_DIR.glob("*.py"))), _hf_modules)

            # Monkey-patch hf_hub_download so conch_v1_5.py finds weights
            # locally. It calls hf_hub_download("MahmoodLab/TITAN",
            # "conch_v1_5_pytorch_model.bin") which fails in offline mode
            # even though the file lives at MODELS_TITAN_DIR.
            import huggingface_hub as _hfh
            import sys as _sys
            if not hasattr(_hfh, "_orig_hf_hub_download"):
                _hfh._orig_hf_hub_download = _hfh.hf_hub_download
            def _patched_hf_hub_dl(repo_id, filename, *args, **kwargs):
                _lp = MODELS_TITAN_DIR / filename
                if _lp.exists():
                    logger.info("hf_hub_download intercepted → %s", _lp)
                    return str(_lp)
                return _hfh._orig_hf_hub_download(repo_id, filename, *args, **kwargs)
            _hfh.hf_hub_download = _patched_hf_hub_dl
            # Evict cached titan modules so fresh import picks up the patch
            for _k in list(_sys.modules.keys()):
                if "transformers_modules" in _k and "titan" in _k.lower():
                    del _sys.modules[_k]

            # Load the TITAN slide-level model
            titan = AutoModel.from_pretrained(
                str(MODELS_TITAN_DIR),
                trust_remote_code=True,
                local_files_only=True,
            )
            titan.eval()
            titan.to(self._device)

            # TITAN is a slide-level model. For patch-level feature extraction
            # we use the CONCH visual encoder that is bundled inside TITAN.
            # titan.return_conch() returns (conch_model, eval_transform).
            conch_model, conch_transform = titan.return_conch()
            self._model = conch_model.to(self._device).eval()
            # Override transform with CONCH's own transform (more accurate than
            # the generic ImageNet one we built in _build_transform).
            self._transform = conch_transform

            # Probe feature dimension with a dummy forward pass through CONCH
            dummy = torch.zeros(1, 3, self._patch_size_px, self._patch_size_px).to(self._device)
            with torch.no_grad():
                out = self._model(dummy)
            if isinstance(out, torch.Tensor):
                feat = out
            elif hasattr(out, "last_hidden_state"):
                feat = out.last_hidden_state[:, 0]
            else:
                feat = out[0]
            self._feature_dim = int(feat.shape[-1])
            logger.info(
                "TITAN+CONCH loaded successfully. Feature dim=%d, device=%s",
                self._feature_dim,
                self._device,
            )
        except Exception as exc:
            logger.error("Failed to load TITAN: %s", exc)
            raise

    def _preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        tensors = [self._transform(img) for img in images]
        return torch.stack(tensors).to(self._device)

    @torch.no_grad()
    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        """
        Extract features from a list of PIL images.

        Parameters
        ----------
        images : list[PIL.Image.Image]
            Already-loaded patch images.

        Returns
        -------
        np.ndarray  shape (N, D) float32
        """
        if self._model is None:
            raise RuntimeError("Model not loaded – call load() first.")
        if not images:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        x = self._preprocess(images)
        out = self._model(x)

        if hasattr(out, "last_hidden_state"):
            feats = out.last_hidden_state[:, 0]
        elif isinstance(out, torch.Tensor):
            feats = out
        else:
            feats = out[0]

        return feats.float().cpu().numpy()

    def extract_patches(
        self,
        patch_paths: list[Path],
        desc: str = "Extracting features",
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """
        Extract features from all patches in *patch_paths* using the configured
        batch size.  Returns:
          - features     np.ndarray (N_success, D)
          - names        list[str]   – file names for successful patches
          - failed_names list[str]   – file names that could not be processed
        """
        from app.io_utils import load_patch_image

        all_features: list[np.ndarray] = []
        all_names: list[str] = []
        failed_names: list[str] = []

        batch_images: list[Image.Image] = []
        batch_names: list[str] = []

        def _flush_batch() -> None:
            nonlocal batch_images, batch_names
            if not batch_images:
                return
            try:
                feats = self.extract_batch(batch_images)
                all_features.append(feats)
                all_names.extend(batch_names)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Batch extraction failed for %d patches – %s",
                    len(batch_names),
                    exc,
                )
                failed_names.extend(batch_names)
            finally:
                batch_images = []
                batch_names = []

        for path in tqdm(patch_paths, desc=desc, unit="patch"):
            arr = load_patch_image(path)
            if arr is None:
                failed_names.append(path.name)
                continue
            batch_images.append(Image.fromarray(arr))
            batch_names.append(path.name)

            if len(batch_images) >= self._batch_size:
                _flush_batch()

        _flush_batch()  # final partial batch

        if all_features:
            features = np.vstack(all_features)
        else:
            features = np.empty((0, self.feature_dim), dtype=np.float32)

        return features, all_names, failed_names
