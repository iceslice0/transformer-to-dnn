"""Shared CLI config loading, argument generation, and string parsing helpers."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Type, TypeVar, Union, get_args, get_origin, get_type_hints

import torch

from transformer_surgery.ops import set_surgery_dtype
from transformer_surgery.util import set_default_device


TConfig = TypeVar("TConfig")


def load_dataclass_from_json(
    cls: Type[TConfig],
    json_path: str,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    config_json_path_field: str = "config_json_path",
) -> TConfig:
    """
    Load JSON into dataclass defaults, then apply explicit overrides.

    Unknown JSON keys are ignored so configs can carry fields for other commands without breaking
    a focused loader.
    """
    cfg = cls()
    allowed = {f.name for f in fields(cls)}
    skip = {config_json_path_field}
    ap = os.path.abspath(os.path.expanduser((json_path or "").strip()))
    with open(ap, encoding="utf-8") as f:
        raw = json.load(f)
    kwargs = {k: v for k, v in raw.items() if k in allowed and k not in skip}
    if kwargs:
        cfg = replace(cfg, **kwargs)
    if overrides:
        kwargs = {k: v for k, v in overrides.items() if k in allowed and k not in skip}
        if kwargs:
            cfg = replace(cfg, **kwargs)
    return replace(cfg, **{config_json_path_field: ap})


def cli_overrides_from_namespace(
    args: Any,
    cls: Type[Any],
    *,
    exclude: frozenset[str] = frozenset({"config", "config_json_path"}),
) -> Dict[str, Any]:
    patchable = {f.name for f in fields(cls)} - exclude
    avars = vars(args)
    return {name: avars[name] for name in patchable if name in avars}


def _snake_to_kebab(name: str) -> str:
    return name.replace("_", "-")


def _strip_optional(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def add_dataclass_cli_args(
    parser: argparse.ArgumentParser,
    cls: Type[Any],
    *,
    exclude: frozenset[str] = frozenset({"config_json_path"}),
    field_help: Optional[Dict[str, str]] = None,
) -> None:
    """Add one ``--kebab-case`` argparse flag for each supported dataclass field."""
    field_help = field_help or {}
    hints = get_type_hints(cls)
    for f in fields(cls):
        if f.name in exclude:
            continue
        name = f.name
        flag = f"--{_snake_to_kebab(name)}"
        h = field_help.get(name)
        tp = _strip_optional(hints.get(name, type(f.default)))
        if tp is bool:
            if f.default is True:
                parser.add_argument(flag, dest=name, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help=h)
            else:
                parser.add_argument(flag, dest=name, action="store_true", default=argparse.SUPPRESS, help=h)
        elif tp is int:
            parser.add_argument(flag, dest=name, type=int, default=argparse.SUPPRESS, help=h)
        elif tp is float:
            parser.add_argument(flag, dest=name, type=float, default=argparse.SUPPRESS, help=h)
        elif tp is str:
            parser.add_argument(flag, dest=name, type=str, default=argparse.SUPPRESS, help=h)
        elif get_origin(tp) is list:
            args = get_args(tp)
            item_type = args[0] if args else str
            if item_type is not str:
                raise TypeError(f"Unsupported CLI list item type {cls.__name__}.{name}: {item_type}")
            parser.add_argument(flag, dest=name, nargs="*", type=str, default=argparse.SUPPRESS, help=h)
        else:
            raise TypeError(f"Unsupported CLI field type {cls.__name__}.{name}: {tp}")


def build_config_cli_parser(
    description: str,
    config_cls: Type[Any],
    *,
    config_default: str,
    config_help: str = "JSON hyperparameters (merged with dataclass defaults).",
    config_dest: str = "config",
    field_help: Optional[Dict[str, str]] = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", dest=config_dest, type=str, default=config_default, help=config_help)
    add_dataclass_cli_args(parser, config_cls, field_help=field_help)
    return parser


def parse_cli_config(
    config_cls: Type[TConfig],
    *,
    description: str,
    config_default: str,
    config_help: str = "JSON hyperparameters (merged with dataclass defaults).",
    field_help: Optional[Dict[str, str]] = None,
    argv: Optional[Sequence[str]] = None,
) -> TConfig:
    parser = build_config_cli_parser(
        description,
        config_cls,
        config_default=config_default,
        config_help=config_help,
        field_help=field_help,
    )
    args = parser.parse_args(argv)
    return config_cls.load(args.config, cli_overrides_from_namespace(args, config_cls))


def torch_dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).strip().replace("torch.", "")
    try:
        value = getattr(torch, normalized)
    except AttributeError as exc:
        raise ValueError(f"Unknown torch dtype {name!r}") from exc
    if not isinstance(value, torch.dtype):
        raise ValueError(f"torch.{normalized} is not a dtype")
    return value


def device_from_config(cfg: Union[Mapping[str, Any], Any]) -> torch.device:
    name = str(cfg.get("device", "cuda")) if isinstance(cfg, Mapping) else str(cfg.device)
    return torch.device(name.strip())


def apply_device_from_config(cfg: Union[Mapping[str, Any], Any]) -> torch.device:
    return set_default_device(device_from_config(cfg))


def surgery_dtype_from_config(cfg: Union[Mapping[str, Any], Any]) -> torch.dtype:
    if isinstance(cfg, Mapping):
        name = str(cfg.get("surgery_dtype", "bfloat16"))
    else:
        name = str(getattr(cfg, "surgery_dtype", "bfloat16"))
    return torch_dtype_from_name(name)


def apply_dtype_from_config(cfg: Union[Mapping[str, Any], Any]) -> torch.dtype:
    dt = surgery_dtype_from_config(cfg)
    set_surgery_dtype(dt)
    return dt
