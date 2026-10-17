"""
Procedural PTQ runner for surgery checkpoints.

Flow: validate the float model -> (optional output identity) -> pick nodes -> gather input ranges -> (optional Brent
scaling, ranges again) -> gather dequant moments -> install wrappers -> save, reload and validate. The float model is the
reference and keeps its surgery dtype (bf16 under autocast) throughout calibration; the quantized accumulations of
calibration and the whole quantized model run in fp32 with autocast off (``ops.ptq_check_fp32``).
"""

from __future__ import annotations

import copy
import itertools
import math
import os
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple, get_args

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_surgery.cli.ptq_config import BrentSite
from transformer_surgery.internal.calibration import (
    ActivationGroup,
    PTQKind,
    PTQNodeMeta,
    PTQNodeStats,
    PTQReloadConfig,
    build_ptq_node_setup,
    build_ptq_wrapper,
    check_ptq_integer_sums,
    gather_ptq_dequant_moments,
    gather_ptq_range_moments,
    ptq_activation_bits,
    ptq_activation_group,
    ptq_module_kind,
    ptq_wrapper_from_reload_config,
)
from transformer_surgery.internal.reporting import (
    describe_device,
    describe_dtype,
    log_line,
    log_wrote,
    metadata_path_for_checkpoint,
    traceable_artifact_path,
    traceable_log_path,
    write_json,
    write_model_structure_txt,
)
from transformer_surgery.internal.util import (
    accuracy_and_loss,
    dataclass_from_record,
    get_device,
    get_surgery_dtype,
    save_model_checkpoint,
    set_seed,
    set_surgery_dtype,
)
from transformer_surgery.models.adapters import (
    build_surgery_student,
    get_model_adapter,
    load_surgery_student_checkpoint,
)
from transformer_surgery.ops import OutputIdentity, RewrittenLayerNorm, SparseWeightedSumBySquare, set_inplace_eval

if TYPE_CHECKING:
    from transformer_surgery.cli.ptq_config import PTQSurgeryConfig


@dataclass
class PTQSelection:
    wrap_linear_conv: bool
    wrap_affine: bool
    wrap_matmul: bool
    include_names: List[str]
    exclude_names: List[str]
    selected_nodes: List[str]


@dataclass
class PTQQuantization:
    weight_bits: int
    activation_bits: int
    compiled_affine_input_bits: int
    matmul_activation_bits: int
    per_output_channel: bool
    dequant_var_eps: float
    quantize_output: bool
    output_bits: Optional[int]
    simulation_dtype: Literal["float32"] = "float32"


@dataclass
class PTQCalibration:
    """Calibration batches and, with ``brent_scaling``, the folded ``lam`` per site and the folded float accuracy."""

    calibration_seed: Optional[int]
    brent_scaling: bool
    loader_split: Literal["train"] = "train"
    mode: Literal["full_train_loader_aggregated"] = "full_train_loader_aggregated"
    brent_sites: Optional[List[BrentSite]] = None
    lambda_by_site: Optional[Dict[str, float]] = None
    fp_val_acc_folded: Optional[float] = None
    fp_val_loss_folded: Optional[float] = None


@dataclass
class PTQMetrics:
    """``fp_*``: the float reference model in its surgery dtype (bf16); deltas are against it."""

    fp_val_acc: float
    fp_val_loss: float
    ptq_val_acc: float
    ptq_val_loss: float
    # With quantize_output: the PTQ model with the output identity bypassed, so the model output is not quantized.
    ptq_val_acc_unquantized_output: Optional[float] = None
    ptq_val_loss_unquantized_output: Optional[float] = None
    acc_delta: float = field(init=False)
    loss_delta: float = field(init=False)

    def __post_init__(self) -> None:
        self.acc_delta = self.ptq_val_acc - self.fp_val_acc
        self.loss_delta = self.ptq_val_loss - self.fp_val_loss


@dataclass
class PTQSummary:
    """PTQ metadata JSON of one run."""

    source_checkpoint: str
    output_checkpoint: str
    selection: PTQSelection
    quantization: PTQQuantization
    calibration: PTQCalibration
    metrics: PTQMetrics
    ptq_nodes: List[PTQNodeMeta]


def _name_matches(name: str, patterns: Sequence[str]) -> bool:
    return any((needle := pattern.strip()) and needle in name for pattern in patterns)


def _build_node_selection(model: nn.Module, cfg: "PTQSurgeryConfig") -> Dict[str, PTQKind]:
    """PTQ kind of each selected node: by group switch or ``include_names``, minus ``exclude_names``."""
    wrap: Dict[ActivationGroup, bool] = {
        "linear": cfg.wrap_linear_conv,
        "affine": cfg.wrap_affine,
        "matmul": cfg.wrap_matmul,
        "output": cfg.quantize_output,
    }
    selected: Dict[str, PTQKind] = {}
    for name, module in model.named_modules():
        kind = ptq_module_kind(module)
        if not name or kind is None or _name_matches(name, cfg.exclude_names):
            continue
        if wrap[ptq_activation_group(kind)] or _name_matches(name, cfg.include_names):
            selected[name] = kind
    return selected


def _filter_unobserved_nodes(
    selected: Dict[str, PTQKind], stats: Dict[str, PTQNodeStats]
) -> Dict[str, PTQKind]:
    """
    Drop selected nodes that never emitted calibration outputs.

    Some modules are conditionally inactive on a given model/config path (for example,
    Gibbs tail-scaling when ``top_k == seq_len``). Those modules are skipped
    rather than failing the full PTQ run.
    """
    skipped = [name for name in selected if stats[name].examples_total == 0]
    if skipped:
        log_line("Skipping PTQ node(s) with no calibration observations: " + ", ".join(skipped))
    return {name: kind for name, kind in selected.items() if name not in skipped}


def _validate(model: nn.Module, loader: Any) -> Tuple[float, float]:
    """Top-1 accuracy and mean cross-entropy (computed in float32)."""
    return accuracy_and_loss(model, loader, lambda logits, y: F.cross_entropy(logits.float(), y))


def attach_output_identity(model: nn.Module) -> None:
    """Route the model output through an :class:`OutputIdentity` node ``ptq_output``, so PTQ can quantize it."""
    model.ptq_output = OutputIdentity().to(get_device())
    model.register_forward_hook(lambda m, _args, out: m.ptq_output(out))


def to_fp32_simulation(model: nn.Module) -> nn.Module:
    """The quantized model in fp32 with autocast off for the rest of the run (surgery dtype float32). Only the PTQ model
    moves to fp32; the float reference keeps its surgery dtype."""
    set_surgery_dtype(torch.float32)
    model = model.float()
    narrow = [n for n, t in itertools.chain(model.named_parameters(), model.named_buffers())
              if t.is_floating_point() and t.dtype != torch.float32]
    if narrow:
        raise RuntimeError(f"PTQ model tensors not in fp32: {narrow[:5]}")
    return model


def load_ptq_checkpoint(path: str, cfg: Any) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a PTQ checkpoint: rebuild the surgery template in its surgery dtype, add the output identity when the run
    quantized the model output, install skeleton PTQ modules, move to fp32, then load_state_dict."""
    set_inplace_eval(False)
    payload = torch.load(path, map_location=get_device(), weights_only=False)
    model, extra = build_surgery_student(payload["extra"], cfg)
    if extra.get("ptq_output_identity", False):
        attach_output_identity(model)
    for record in extra["ptq_wrappers"]:
        reload_config = dataclass_from_record(PTQReloadConfig, record)
        wrapper = ptq_wrapper_from_reload_config(reload_config, model.get_submodule(reload_config.name))
        model.set_submodule(reload_config.name, wrapper.to(get_device()))
    model = to_fp32_simulation(model)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, extra


def _power_of_two(x: float) -> float:
    return 2.0 ** round(math.log2(x))


@torch.no_grad()
def apply_brent_scaling(
    model: nn.Module, stats: Dict[str, PTQNodeStats], sites: Sequence[BrentSite] = get_args(BrentSite)
) -> Dict[str, float]:
    """Fold Brent's operand scaling into the norm-expansion products of the strict graph, in place.

    Mix ``sum_k p v``: ``v -> v / lam``, ``p -> lam p`` with ``lam^2 = max|v| / max|p|``, so ``sub_vp`` scales its
    inputs by ``[1/lam, lam]``, ``diff_sum`` its ``v^2`` input by ``1/lam^2`` and ``combine`` its ``sum p^2`` input by
    ``lam^2``. LayerNorm ``u * a``: ``u -> lam u``, ``a -> a / lam`` with ``lam^2 = max|a| / max|u|``, so
    ``add_u_invstd`` scales by ``[lam, 1/lam]`` and ``combine_z`` its ``u^2`` and ``a^2`` inputs by ``lam^2`` and
    ``1/lam^2``. The ranges come from the first calibration pass (inputs of ``sub_vp`` and ``add_u_invstd``); ``lam`` is
    rounded to a power of two, so every folded coefficient is exact in bf16 and the graph computes the same function.
    """
    unknown = set(sites) - set(get_args(BrentSite))
    if unknown:
        raise ValueError(f"unknown brent_sites {sorted(unknown)}; use any of {get_args(BrentSite)}")
    lams: Dict[str, float] = {}

    def fold(node: nn.Module, factors: Sequence[float]) -> None:
        node.coeff.mul_(torch.tensor(factors, dtype=node.coeff.dtype, device=node.coeff.device))

    for name, m in model.named_modules():
        if isinstance(m, SparseWeightedSumBySquare) and m.norm_expansion and "mix" in sites:
            mv, mp = (float(t.max()) for t in stats[f"{name}.sub_vp"].input_max_abs[:2])
            lam = _power_of_two(math.sqrt(mv / mp))
            fold(m.sub_vp, [1.0 / lam, lam])
            fold(m.diff_sum, [1.0 / lam**2, 1.0])
            fold(m.combine, [1.0, lam**2])
        elif isinstance(m, RewrittenLayerNorm) and m.norm_ln and "layernorm" in sites:
            mu, ma = (float(t.max()) for t in stats[f"{name}.add_u_invstd"].input_max_abs[:2])
            lam = _power_of_two(math.sqrt(ma / mu))
            fold(m.add_u_invstd, [lam, 1.0 / lam])
            fold(m.combine_z, [1.0, lam**2, 1.0 / lam**2])
        else:
            continue
        lams[name] = lam
    return lams


def run_ptq(cfg: "PTQSurgeryConfig") -> None:
    # PTQ calibration reads node inputs in forward hooks after the node has run, so no node may overwrite them.
    set_inplace_eval(False)
    out_abs = traceable_artifact_path(cfg.output, cfg, "ts-ptq", extension=".pt")
    meta_abs = metadata_path_for_checkpoint(out_abs)
    model_log_abs = traceable_log_path(cfg.log_dir, cfg, "ts-ptq", "model_after_ptq")
    cfg.output = out_abs

    log_line(f"Using device: {describe_device(get_device())}")
    log_line(f"config_json={cfg.config_json_path}")
    fp_model, fp_extra = load_surgery_student_checkpoint(os.path.abspath(cfg.fp_checkpoint), cfg)
    log_line(f"Surgery dtype (from checkpoint): {describe_dtype(get_surgery_dtype())}")
    adapter = get_model_adapter(fp_extra["model_key"])
    if cfg.calibration_seed is not None:
        set_seed(cfg.calibration_seed)
    log_line(f"Calibration seed: {cfg.calibration_seed}")
    train_loader, val_loader = adapter.build_loaders(cfg)
    log_line(f"Using model adapter: {adapter.key}")

    calibration_batches = list(range(len(train_loader)))
    if not calibration_batches:
        raise SystemExit("No training batches available for PTQ calibration.")
    log_line(f"PTQ calibration batches: full train loader ({len(calibration_batches)} batch(es), aggregated stats).")

    fp_acc, fp_loss = _validate(fp_model, val_loader)
    log_line(f"Float model ({describe_dtype(get_surgery_dtype())}) val acc={fp_acc:.4f} loss={fp_loss:.4f}")
    if cfg.quantize_output:
        attach_output_identity(fp_model)
        log_line("Model output routed through ptq_output (OutputIdentity), so PTQ quantizes it.")

    selected = _build_node_selection(fp_model, cfg)
    if not selected:
        raise SystemExit("No PTQ-wrappable nodes selected by the current config.")
    log_line(f"Selected {len(selected)} PTQ node(s).")
    for name, kind in selected.items():
        log_line(f"  {name}: {kind}")

    stats = gather_ptq_range_moments(fp_model, train_loader, selected, calibration_batches)
    selected = _filter_unobserved_nodes(selected, stats)
    if not selected:
        raise SystemExit("No PTQ node received calibration statistics on the current run.")
    calibration = PTQCalibration(calibration_seed=cfg.calibration_seed, brent_scaling=cfg.brent_scaling)
    if cfg.brent_scaling:
        lams = apply_brent_scaling(fp_model, stats, cfg.brent_sites)
        if not lams:
            raise SystemExit("brent_scaling=true but the model has no norm-expansion Mix or LayerNorm.")
        log_line("Brent scaling lam by site: " + ", ".join(f"{n}={v:g}" for n, v in lams.items()))
        folded_acc, folded_loss = _validate(fp_model, val_loader)
        log_line(f"Float model with folded lam val acc={folded_acc:.4f} loss={folded_loss:.4f}")
        calibration.brent_sites = list(cfg.brent_sites)
        calibration.lambda_by_site = lams
        calibration.fp_val_acc_folded, calibration.fp_val_loss_folded = folded_acc, folded_loss
        # The folded coefficients change the ranges of the nodes downstream of a +- b: measure them again.
        stats = gather_ptq_range_moments(fp_model, train_loader, selected, calibration_batches)
        selected = _filter_unobserved_nodes(selected, stats)

    setups = {name: build_ptq_node_setup(name, fp_model.get_submodule(name), stats[name], cfg) for name in selected}
    gather_ptq_dequant_moments(fp_model, train_loader, selected, stats, setups, calibration_batches)
    missing_dequant = [n for n in selected if stats[n].bias_fit.count <= 0 and stats[n].affine_fit.count <= 0]
    if missing_dequant:
        raise SystemExit(f"Calibration dequant statistics missing for selected node(s): {missing_dequant}")
    check_ptq_integer_sums({n: stats[n] for n in selected})
    bound_name = max(selected, key=lambda n: stats[n].integer_sum_bound or 0.0)
    log_line(f"Largest integer sum bound: {stats[bound_name].integer_sum_bound:.0f} at {bound_name} (fp32 exact to 2^24)")

    # Only the quantized model moves to fp32; calibration above used the float reference in its surgery dtype.
    ptq_model = to_fp32_simulation(copy.deepcopy(fp_model))
    node_meta: List[PTQNodeMeta] = []
    reload_configs: List[PTQReloadConfig] = []
    for name in selected:
        module = ptq_model.get_submodule(name)
        built = build_ptq_wrapper(module, setups[name], stats[name], dequant_var_eps=cfg.dequant_var_eps)
        ptq_model.set_submodule(name, built.module)
        node_meta.append(built.metadata)
        reload_configs.append(built.reload_config)

    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    save_model_checkpoint(
        out_abs,
        ptq_model,
        extra={
            **fp_extra,
            "ptq_meta_path": os.path.basename(meta_abs),
            "ptq_wrappers": [asdict(c) for c in reload_configs],
            "ptq_output_identity": bool(cfg.quantize_output),
            "ptq_simulation_dtype": "float32",
        },
    )
    write_model_structure_txt(model_log_abs, ptq_model, "PTQ Surgery Model")
    log_wrote(model_log_abs)
    log_wrote(out_abs)

    del ptq_model
    reloaded_model, _ = load_ptq_checkpoint(out_abs, cfg)
    log_line("reloaded PTQ checkpoint from disk for validation")
    ptq_acc, ptq_loss = _validate(reloaded_model, val_loader)
    log_line(f"PTQ model val acc={ptq_acc:.4f} loss={ptq_loss:.4f}")
    log_line(f"Delta acc={ptq_acc - fp_acc:+.4f} loss={ptq_loss - fp_loss:+.4f}")
    open_acc = open_loss = None
    if cfg.quantize_output:
        output_wrapper = reloaded_model.ptq_output
        reloaded_model.ptq_output = nn.Identity()
        open_acc, open_loss = _validate(reloaded_model, val_loader)
        reloaded_model.ptq_output = output_wrapper
        log_line(f"PTQ model with the output not quantized val acc={open_acc:.4f} loss={open_loss:.4f}")

    summary = PTQSummary(
        source_checkpoint=os.path.abspath(cfg.fp_checkpoint),
        output_checkpoint=out_abs,
        selection=PTQSelection(
            wrap_linear_conv=cfg.wrap_linear_conv,
            wrap_affine=cfg.wrap_affine,
            wrap_matmul=cfg.wrap_matmul,
            include_names=list(cfg.include_names),
            exclude_names=list(cfg.exclude_names),
            selected_nodes=list(selected),
        ),
        quantization=PTQQuantization(
            weight_bits=cfg.weight_bits,
            activation_bits=ptq_activation_bits("linear", cfg),
            compiled_affine_input_bits=ptq_activation_bits("affine_scale", cfg),
            matmul_activation_bits=ptq_activation_bits("matmul", cfg),
            per_output_channel=cfg.per_output_channel,
            dequant_var_eps=cfg.dequant_var_eps,
            quantize_output=cfg.quantize_output,
            output_bits=ptq_activation_bits("output_identity", cfg) if cfg.quantize_output else None,
        ),
        calibration=calibration,
        metrics=PTQMetrics(
            fp_val_acc=fp_acc,
            fp_val_loss=fp_loss,
            ptq_val_acc=ptq_acc,
            ptq_val_loss=ptq_loss,
            ptq_val_acc_unquantized_output=open_acc,
            ptq_val_loss_unquantized_output=open_loss,
        ),
        ptq_nodes=node_meta,
    )
    write_json(meta_abs, asdict(summary))
    log_wrote(meta_abs)
