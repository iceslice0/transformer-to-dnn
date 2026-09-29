"""WikiText-2 fixed-window loaders for Pythia surgery-only evaluation."""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


def _require_llm_deps() -> None:
    try:
        import datasets  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Pythia/WikiText-2 support requires optional deps. Install with: "
            "pip install -e '.[llm]'"
        ) from exc


def _concat_split_text(rows: Sequence[dict], text_key: str = "text") -> str:
    parts: List[str] = []
    for row in rows:
        text = row.get(text_key, "")
        if text is None:
            continue
        s = str(text).strip()
        if s:
            parts.append(s)
    return "\n\n".join(parts)


def tokenize_and_window(
    text: str,
    *,
    tokenizer: Any,
    context_length: int,
    max_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Emit non-overlapping windows of length ``context_length``.

    Each window uses ``context_length + 1`` tokens from the stream so
    ``input_ids`` / ``labels`` are next-token aligned and every token (except
    the final incomplete window) is predicted exactly once.
    """
    ctx = int(context_length)
    if ctx < 1:
        raise ValueError(f"context_length must be >= 1, got {ctx}")
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    ids = list(encoded["input_ids"])
    if max_tokens is not None:
        ids = ids[: int(max_tokens)]
    inputs: List[List[int]] = []
    labels: List[List[int]] = []
    step = ctx
    need = ctx + 1
    for start in range(0, len(ids) - ctx, step):
        chunk = ids[start : start + need]
        if len(chunk) < need:
            break
        inputs.append(chunk[:-1])
        labels.append(chunk[1:])
    if not inputs:
        raise ValueError(
            f"Not enough tokens ({len(ids)}) for context_length={ctx}; "
            "need at least context_length+1 tokens after tokenization."
        )
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class CausalLMWindowDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, labels: torch.Tensor) -> None:
        if input_ids.shape != labels.shape:
            raise ValueError(f"input/label shape mismatch: {tuple(input_ids.shape)} vs {tuple(labels.shape)}")
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.labels[idx]


def build_wikitext2_loaders(cfg: Any) -> Tuple[DataLoader, DataLoader]:
    """Download/load WikiText-2, tokenize with the HF tokenizer, return train/val window loaders."""
    _require_llm_deps()
    from datasets import load_dataset
    from transformers import AutoTokenizer

    data_dir = os.path.abspath(str(getattr(cfg, "data_dir", "./data")))
    os.makedirs(data_dir, exist_ok=True)
    cache_dir = os.path.join(data_dir, "hf_datasets")
    model_id = str(getattr(cfg, "hf_model_id", "EleutherAI/pythia-70m"))
    ds_name = str(getattr(cfg, "dataset_name", "wikitext"))
    ds_config = str(getattr(cfg, "dataset_config", "wikitext-2-raw-v1"))
    context_length = int(getattr(cfg, "context_length", 128))
    batch_size = int(getattr(cfg, "batch_size", 8))
    workers = int(getattr(cfg, "workers", 0))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    raw = load_dataset(ds_name, ds_config, cache_dir=cache_dir)
    train_text = _concat_split_text(raw["train"])
    val_text = _concat_split_text(raw["validation"])

    train_x, train_y = tokenize_and_window(
        train_text,
        tokenizer=tokenizer,
        context_length=context_length,
        max_tokens=getattr(cfg, "max_train_tokens", None),
    )
    val_x, val_y = tokenize_and_window(
        val_text,
        tokenizer=tokenizer,
        context_length=context_length,
        max_tokens=getattr(cfg, "max_val_tokens", None),
    )

    train_ds = CausalLMWindowDataset(train_x, train_y)
    val_ds = CausalLMWindowDataset(val_x, val_y)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
