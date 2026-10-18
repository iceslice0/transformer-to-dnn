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
from torchvision.transforms.functional import to_tensor

from transformer_surgery.internal.util import make_loaders

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        return to_tensor(img.convert("RGB"))


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
        center_crop: bool = False,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.train = train
        self.center_crop = bool(center_crop)
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
        elif self.center_crop:
            hr, lr = self._paired_center_crop(hr, lr)
        return lr, hr

    def _paired_center_crop(self, hr: torch.Tensor, lr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, h_lr, w_lr = lr.shape
        ps, s = self.patch_size_lq, self.scale
        top, left = (h_lr - ps) // 2, (w_lr - ps) // 2
        return hr[:, top * s:(top + ps) * s, left * s:(left + ps) * s], lr[:, top:top + ps, left:left + ps]


def build_mambair_loaders(cfg: Any) -> Tuple[DataLoader, DataLoader]:
    """Train/val paired-SR loaders. Dataset dirs come from explicit cfg fields or, when absent,
    canonical paths under ``cfg.data_dir``; missing datasets are downloaded when ``sr_download``."""
    from transformer_surgery.models.mambair.download import default_sr_paths, ensure_sr_datasets

    defaults = default_sr_paths(cfg.data_dir, cfg.scale)
    paths = {key: getattr(cfg, key) or default for key, default in defaults.items()}
    missing = [key for key, path in paths.items() if not os.path.isdir(path)]
    if missing:
        if not cfg.sr_download:
            raise FileNotFoundError(
                f"MambaIR SR dataset dirs missing: {', '.join(paths[k] for k in missing)} "
                f"(set sr_download=true to fetch, or point train_hr/train_lr/val_hr/val_lr at existing data)"
            )
        where = os.path.abspath(cfg.data_dir)
        print(f"MambaIR SR datasets missing ({', '.join(missing)}); downloading into {where} ...", flush=True)
        paths = ensure_sr_datasets(cfg.data_dir, cfg.scale)

    train_ds = PairedSRDataset(
        paths["train_hr"],
        paths["train_lr"],
        scale=cfg.scale,
        train=True,
        patch_size_lq=cfg.patch_size,
        repeat=cfg.train_repeat,
        max_items=cfg.max_train_items,
    )
    val_ds = PairedSRDataset(
        paths["val_hr"], paths["val_lr"], scale=cfg.scale, train=False, max_items=cfg.max_val_items
    )
    return make_loaders(train_ds, val_ds, cfg, val_batch_size=1)


def build_mambair_selection_loader(cfg: Any) -> DataLoader:
    """The DIV2K training pairs as one centered training-size patch each, without augmentation and in a fixed order:
    the data for checkpoint selection. Set5 is evaluated only."""
    from transformer_surgery.models.mambair.download import default_sr_paths

    defaults = default_sr_paths(cfg.data_dir, cfg.scale)
    paths = {key: getattr(cfg, key) or default for key, default in defaults.items()}
    select_ds = PairedSRDataset(
        paths["train_hr"],
        paths["train_lr"],
        scale=cfg.scale,
        train=False,
        patch_size_lq=cfg.patch_size,
        max_items=cfg.max_train_items,
        center_crop=True,
    )
    return DataLoader(select_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)
