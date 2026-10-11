"""Shared run-config base classes, JSON loading, and argparse generation for the CLIs."""

from __future__ import annotations

import argparse
import functools
import json
import os
from dataclasses import dataclass, fields
from typing import Any, Dict, Literal, Optional, Sequence, Type, TypeVar, Union, get_args, get_origin, get_type_hints

import torch

from transformer_surgery.internal.util import (
    DEFAULT_MODEL_KEY,
    dataclass_from_record,
    set_default_device,
    set_surgery_dtype,
    torch_dtype_from_name,
)

TConfig = TypeVar("TConfig", bound="RunConfig")


@dataclass
class RunConfig:
    """Data loading and device fields of every stage; JSON + CLI via :meth:`load`."""

    data_dir: str = "./data"
    batch_size: int = 128
    workers: int = 2
    randaugment: bool = True
    ra_magnitude: int = 9
    random_erasing_prob: float = 0.0
    device: str = "cuda"
    config_json_path: Optional[str] = None

    @classmethod
    def load(cls: Type[TConfig], json_path: str, overrides: Optional[Dict[str, Any]] = None) -> TConfig:
        """JSON values over the dataclass defaults, then ``overrides``; keys of other configs are ignored."""
        path = os.path.abspath(os.path.expanduser(json_path.strip()))
        with open(path, encoding="utf-8") as f:
            values = {**json.load(f), **(overrides or {})}
        return dataclass_from_record(cls, {**values, "config_json_path": path})


@dataclass
class AdapterRunConfig(RunConfig):
    """Stages that run through a model adapter (surgery, distill, PTQ)."""

    model_key: str = DEFAULT_MODEL_KEY
    reference_checkpoint: Optional[str] = None
    # Super-resolution (MambaIR) fields; ignored by classification adapters.
    scale: int = 2
    train_hr: Optional[str] = None
    train_lr: Optional[str] = None
    val_hr: Optional[str] = None
    val_lr: Optional[str] = None
    patch_size: int = 64
    train_repeat: int = 1
    max_train_items: Optional[int] = None
    max_val_items: Optional[int] = None
    sr_download: bool = True


FIELD_HELP_COMMON: Dict[str, str] = {
    "model_key": f"Model adapter key. Default: {DEFAULT_MODEL_KEY}.",
    "reference_checkpoint": "Reference/teacher checkpoint path.",
}


def _strip_optional(tp: Any) -> Any:
    if get_origin(tp) is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def parse_cli_config(
    config_cls: Type[TConfig],
    *,
    description: str,
    config_default: str,
    field_help: Optional[Dict[str, str]] = None,
    argv: Optional[Sequence[str]] = None,
) -> TConfig:
    """``--config`` JSON plus one ``--kebab-case`` flag per dataclass field; flags given override the JSON."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", type=str, default=config_default, help=f"JSON hyperparameters over {config_cls.__name__} defaults."
    )
    helps = {**FIELD_HELP_COMMON, **(field_help or {})}
    hints = get_type_hints(config_cls)
    for f in fields(config_cls):
        if f.name == "config_json_path":
            continue
        add = functools.partial(
            parser.add_argument,
            "--" + f.name.replace("_", "-"),
            dest=f.name,
            default=argparse.SUPPRESS,
            help=helps.get(f.name),
        )
        tp = _strip_optional(hints[f.name])
        item = get_args(tp)[0] if get_origin(tp) is list else None
        if tp is bool:
            add(action=argparse.BooleanOptionalAction if f.default else "store_true")
        elif tp in (int, float, str):
            add(type=tp)
        elif get_origin(tp) is Literal:
            add(choices=get_args(tp))
        elif item is str:
            add(nargs="*", type=str)
        elif get_origin(item) is Literal:
            add(nargs="*", choices=get_args(item))
        else:
            raise TypeError(f"Unsupported CLI field type {config_cls.__name__}.{f.name}: {tp}")
    args = vars(parser.parse_args(argv))
    return config_cls.load(args.pop("config"), args)


def apply_device_from_config(cfg: Any) -> torch.device:
    return set_default_device(torch.device(str(cfg.device).strip()))


def apply_dtype_from_config(cfg: Any) -> None:
    set_surgery_dtype(torch_dtype_from_name(cfg.surgery_dtype))
