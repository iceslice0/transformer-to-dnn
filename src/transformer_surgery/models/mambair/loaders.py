"""Paired HR/LR super-resolution loaders for MambaIR (DIV2K train / Set5 val).

Ported from ``mambal/softmax/softmax_surgery.py`` (``PairedSRDataset`` / ``SRDataModule``).
Datasets yield ``(lq, hr)`` tensor tuples so the framework's ``for x, y in loader`` training
loops work unchanged: ``x`` is the low-res input, ``y`` is the HR target.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from transformer_surgery.internal.util import get_device

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    n_channels = len(img.getbands())
    x = x.view(img.size[1], img.size[0], n_channels)
    x = x.permute(2, 0, 1).contiguous().float() / 255.0
    return x


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        return pil_to_tensor(img)


def canonical_sr_stem(path: Path, scale: int) -> str:
    stem = path.stem
    patterns = [
        rf"_LRBI_x{scale}$",
        rf"_LRBIX{scale}$",
        rf"_LRbic_x{scale}$",
        rf"_lr_x{scale}$",
        rf"_x{scale}$",
        rf"x{scale}$",
    ]
    for pat in patterns:
        stem = re.sub(pat, "", stem, flags=re.IGNORECASE)
    return stem


def match_paired_images(hr_dir: Path, lr_dir: Path, scale: int) -> List[Tuple[Path, Path]]:
    hr_files = sorted([p for p in hr_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    lr_files = sorted([p for p in lr_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])

    hr_map = {canonical_sr_stem(p, scale): p for p in hr_files}
    lr_map = {canonical_sr_stem(p, scale): p for p in lr_files}

    common = sorted(set(hr_map).intersection(lr_map))
    if not common:
        inferred_scales = sorted(
            {
                int(m.group(1))
                for p in lr_files
                for m in [re.search(r"x(\d+)$", p.stem, flags=re.IGNORECASE)]
                if m is not None
            }
        )
        hint = ""
        if inferred_scales and scale not in inferred_scales:
            hint = f" Hint: LR filenames look like x{inferred_scales}, but scale={scale}."
        raise RuntimeError(f"No matched HR/LR image pairs under {hr_dir} and {lr_dir}.{hint}")
    return [(hr_map[k], lr_map[k]) for k in common]


class PairedSRDataset(Dataset):
    """HR/LR pairs matched by canonical stem; returns ``(lq, hr)``."""

    def __init__(
        self,
        hr_dir: str,
        lr_dir: str,
        scale: int,
        train: bool,
        patch_size_lq: int = 64,
        repeat: int = 1,
        max_items: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.train = train
        self.patch_size_lq = patch_size_lq
        self.repeat = max(1, repeat)
        pairs = match_paired_images(Path(hr_dir), Path(lr_dir), scale)
        if max_items is not None:
            pairs = pairs[:max_items]
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs) * self.repeat if self.train else len(self.pairs)

    def _paired_random_crop(self, hr: torch.Tensor, lr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, h_lr, w_lr = lr.shape
        ps = self.patch_size_lq
        if h_lr < ps or w_lr < ps:
            raise ValueError(f"LR patch {ps} is larger than image size {(h_lr, w_lr)}")
        top = random.randint(0, h_lr - ps)
        left = random.randint(0, w_lr - ps)
        lr_crop = lr[:, top:top + ps, left:left + ps]

        s = self.scale
        top_hr = top * s
        left_hr = left * s
        hr_crop = hr[:, top_hr:top_hr + ps * s, left_hr:left_hr + ps * s]
        return hr_crop, lr_crop

    @staticmethod
    def _augment(hr: torch.Tensor, lr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            hr = torch.flip(hr, dims=[2])
            lr = torch.flip(lr, dims=[2])
        if random.random() < 0.5:
            hr = torch.flip(hr, dims=[1])
            lr = torch.flip(lr, dims=[1])
        k = random.randint(0, 3)
        if k:
            hr = torch.rot90(hr, k=k, dims=[1, 2])
            lr = torch.rot90(lr, k=k, dims=[1, 2])
        return hr.contiguous(), lr.contiguous()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr_path, lr_path = self.pairs[idx % len(self.pairs)]
        hr = load_rgb(hr_path)
        lr = load_rgb(lr_path)
        if self.train:
            hr, lr = self._paired_random_crop(hr, lr)
            hr, lr = self._augment(hr, lr)
        return lr, hr


def resolve_sr_paths(cfg: Any) -> dict:
    """HR/LR dirs: explicit cfg fields win, else canonical paths under ``cfg.data_dir``/scale."""
    from transformer_surgery.models.mambair.download import default_sr_paths

    defaults = default_sr_paths(str(cfg.data_dir), int(cfg.scale))
    return {key: (getattr(cfg, key, None) or defaults[key]) for key in defaults}


def build_mambair_loaders(cfg: Any) -> Tuple[DataLoader, DataLoader]:
    """Train/val paired-SR loaders. Dataset dirs come from explicit cfg fields or, when absent,
    canonical paths under ``cfg.data_dir``; missing datasets are downloaded when ``sr_download``."""
    device = get_device()
    scale = int(cfg.scale)
    workers = int(cfg.workers)
    pin = device.type == "cuda"
    persistent = workers > 0
    paths = resolve_sr_paths(cfg)

    missing = [key for key in ("train_hr", "train_lr", "val_hr", "val_lr") if not os.path.isdir(paths[key])]
    if missing:
        if bool(getattr(cfg, "sr_download", True)):
            from transformer_surgery.models.mambair.download import ensure_sr_datasets

            print(f"MambaIR SR datasets missing ({', '.join(missing)}); downloading into {os.path.abspath(cfg.data_dir)} ...", flush=True)
            paths = ensure_sr_datasets(str(cfg.data_dir), scale)
        else:
            raise FileNotFoundError(
                f"MambaIR SR dataset dirs missing: {', '.join(paths[k] for k in missing)} "
                f"(set sr_download=true to fetch, or point train_hr/train_lr/val_hr/val_lr at existing data)"
            )

    train_ds = PairedSRDataset(
        paths["train_hr"],
        paths["train_lr"],
        scale=scale,
        train=True,
        patch_size_lq=int(cfg.patch_size),
        repeat=int(getattr(cfg, "train_repeat", 1)),
        max_items=getattr(cfg, "max_train_items", None),
    )
    val_ds = PairedSRDataset(
        paths["val_hr"],
        paths["val_lr"],
        scale=scale,
        train=False,
        max_items=getattr(cfg, "max_val_items", None),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    return train_loader, val_loader
