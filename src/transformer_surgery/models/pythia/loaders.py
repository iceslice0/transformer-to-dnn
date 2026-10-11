"""WikiText-2 fixed-window loaders for Pythia surgery-only evaluation."""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from transformer_surgery.internal.util import make_loaders


def require_llm_deps() -> None:
    try:
        import datasets  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError("Pythia/WikiText-2 support requires optional deps: pip install -e '.[llm]'") from exc


def _concat_split_text(rows: Sequence[dict]) -> str:
    return "\n\n".join(text for row in rows if (text := str(row.get("text") or "").strip()))


def tokenize_and_window(
    text: str,
    *,
    tokenizer: Any,
    context_length: int,
    max_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Non-overlapping windows of ``context_length`` tokens with next-token labels.

    Window ``i`` reads tokens ``[i*ctx, (i+1)*ctx]``, so every token except those of the final incomplete window is
    predicted exactly once.
    """
    if context_length < 1:
        raise ValueError(f"context_length must be >= 1, got {context_length}")
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"][:max_tokens]
    n = (len(ids) - 1) // context_length
    if n < 1:
        raise ValueError(
            f"Not enough tokens ({len(ids)}) for context_length={context_length}; "
            "need at least context_length+1 tokens after tokenization."
        )
    stream = torch.tensor(ids[: n * context_length + 1], dtype=torch.long)
    return stream[:-1].view(n, context_length), stream[1:].view(n, context_length)


def build_wikitext2_loaders(cfg: Any) -> Tuple[DataLoader, DataLoader]:
    """Download/load WikiText-2, tokenize with the HF tokenizer, return train/val window loaders."""
    require_llm_deps()
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(cfg.data_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model_id)
    cache_dir = os.path.join(os.path.abspath(cfg.data_dir), "hf_datasets")
    raw = load_dataset(cfg.dataset_name, cfg.dataset_config, cache_dir=cache_dir)
    splits = []
    for split, max_tokens in (("train", cfg.max_train_tokens), ("validation", cfg.max_val_tokens)):
        text = _concat_split_text(raw[split])
        x, y = tokenize_and_window(text, tokenizer=tokenizer, context_length=cfg.context_length, max_tokens=max_tokens)
        splits.append(TensorDataset(x, y))
    return make_loaders(*splits, cfg)
