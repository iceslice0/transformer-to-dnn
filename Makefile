.PHONY: all a b c d e a_imagenet mambair mambair-x2 mambair-x4 mambair-x4-exact mambair-x4-tailmass0 mambair-data pythia-smoke pythia pythia-tailmass0 routing pythia-routing routing-strict routing-strict-bench exact-fp64 strict-sr ptq-strict-sr hadamard-sr exact-fp64-hadamard profile-hadamard profile-gs mix-precision strict-pol ptq-balance-check ptq-brent ptq-brent-mix dtype-check help

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

# Main experiment sweeps.
MAIN_KS := 1 2 4 8 16 32 64 96 128 197
TAIL_KS := 1 2 4 8 16 32 64
STRICT_KS := 16 32 64
PTQ_KS := 16 32 64 197
IMAGENET_MAIN_KS := 1 2 4 8 16 32 64 96 128 197

# MambaIRv2 Light SR sweep. k=0 is uniform attention (affine key-mean, not Gibbs);
# k=256 is full window support (window_size=16 -> seq_len=256).
MAMBAIR_KS := 0 1 2 4 16 64 256
MAMBAIR_SCALES := 2 4

SURGERY_MAIN_TARGETS := $(foreach k,$(MAIN_KS),artifacts/checkpoints/ts_surgery_topk$(k)_fast.pt)
IMAGENET_SURGERY_MAIN_TARGETS := $(foreach k,$(IMAGENET_MAIN_KS),artifacts/checkpoints/ts_surgery_topk$(k)_fast_imagenet.pt)
DISTILL_MAIN_CE_TARGETS := $(foreach k,$(MAIN_KS),artifacts/checkpoints/ts_distill_$(k)_fast_ce.pt)
DISTILL_MAIN_J_TARGETS := $(foreach k,$(MAIN_KS),artifacts/checkpoints/ts_distill_$(k)_fast_jeffreys.pt)

TAIL_SURGERY_FIXED0_TARGETS := $(foreach k,$(TAIL_KS),artifacts/checkpoints/ts_surgery_topk$(k)_fast_tailmass0.pt)
TAIL_SURGERY_FIXED1E3_TARGETS := $(foreach k,$(TAIL_KS),artifacts/checkpoints/ts_surgery_topk$(k)_fast_tailmass1e3.pt)
TAIL_SURGERY_CAL_TARGETS := $(foreach k,$(TAIL_KS),artifacts/checkpoints/ts_surgery_topk$(k)_fast.pt)
TAIL_DISTILL_TARGETS := $(foreach k,$(TAIL_KS),artifacts/checkpoints/ts_distill_$(k)_fast_jeffreys.pt)
TAIL_SURGERY_EXACT_TARGETS := $(foreach k,$(MAIN_KS),artifacts/checkpoints/ts_surgery_topk$(k)_fast_tailmass_exact.pt)

STRICT_FAST_TARGETS := $(foreach k,$(STRICT_KS),artifacts/checkpoints/ts_surgery_topk$(k)_fast.pt)
STRICT_STRICT_TARGETS := $(foreach k,$(STRICT_KS),artifacts/checkpoints/ts_surgery_topk$(k)_strict.pt)
IMAGENET_SURGERY_TARGETS := $(IMAGENET_SURGERY_MAIN_TARGETS)

PTQ_FAST_CHANNEL_TARGETS := $(foreach k,$(PTQ_KS),artifacts/checkpoints/ts_ptq_$(k)_fast_jeffreys_8bit.pt)
PTQ_FAST_TENSOR_TARGETS := $(foreach k,$(PTQ_KS),artifacts/checkpoints/ts_ptq_$(k)_fast_jeffreys_pertensor_8bit.pt)
PTQ_STRICT_CHANNEL_TARGETS := $(foreach k,$(PTQ_KS),artifacts/checkpoints/ts_ptq_$(k)_strict_8bit.pt)
PTQ_STRICT_TENSOR_TARGETS := $(foreach k,$(PTQ_KS),artifacts/checkpoints/ts_ptq_$(k)_strict_pertensor_8bit.pt)
PTQ_TARGETS := $(PTQ_FAST_CHANNEL_TARGETS) $(PTQ_FAST_TENSOR_TARGETS) $(PTQ_STRICT_CHANNEL_TARGETS) $(PTQ_STRICT_TENSOR_TARGETS)

MAMBAIR_SURGERY_TARGETS := $(foreach s,$(MAMBAIR_SCALES),$(foreach k,$(MAMBAIR_KS),artifacts/checkpoints/ts_surgery_mambair_x$(s)_topk$(k)_fast.pt))
MAMBAIR_DISTILL_TARGETS := $(foreach s,$(MAMBAIR_SCALES),$(foreach k,$(MAMBAIR_KS),artifacts/checkpoints/ts_distill_mambair_x$(s)_topk$(k).pt))
MAMBAIR_X2_TARGETS := $(foreach k,$(MAMBAIR_KS),artifacts/checkpoints/ts_surgery_mambair_x2_topk$(k)_fast.pt) \
	$(foreach k,$(MAMBAIR_KS),artifacts/checkpoints/ts_distill_mambair_x2_topk$(k).pt)
MAMBAIR_X4_TARGETS := $(foreach k,$(MAMBAIR_KS),artifacts/checkpoints/ts_surgery_mambair_x4_topk$(k)_fast.pt) \
	$(foreach k,$(MAMBAIR_KS),artifacts/checkpoints/ts_distill_mambair_x4_topk$(k).pt)
MAMBAIR_X4_EXACT_SURGERY_TARGETS := $(foreach k,$(MAMBAIR_KS),artifacts/checkpoints/ts_surgery_mambair_x4_topk$(k)_fast_tailmass_exact.pt)
MAMBAIR_X4_EXACT_DISTILL_TARGETS := $(foreach k,$(MAMBAIR_KS),artifacts/checkpoints/ts_distill_mambair_x4_topk$(k)_tailmass_exact.pt)
MAMBAIR_X4_EXACT_TARGETS := $(MAMBAIR_X4_EXACT_SURGERY_TARGETS) $(MAMBAIR_X4_EXACT_DISTILL_TARGETS)
# Tail-drop ablation (q=0, no calibration): k=0 is skipped since an empty support with q=0 has no output.
MAMBAIR_TAIL0_KS := 1 2 4 16 64
MAMBAIR_X4_TAIL0_SURGERY_TARGETS := $(foreach k,$(MAMBAIR_TAIL0_KS),artifacts/checkpoints/ts_surgery_mambair_x4_topk$(k)_fast_tailmass0.pt)
MAMBAIR_X4_TAIL0_DISTILL_TARGETS := $(foreach k,$(MAMBAIR_TAIL0_KS),artifacts/checkpoints/ts_distill_mambair_x4_topk$(k)_tailmass0.pt)
MAMBAIR_X4_TAIL0_TARGETS := $(MAMBAIR_X4_TAIL0_SURGERY_TARGETS) $(MAMBAIR_X4_TAIL0_DISTILL_TARGETS)

# Pythia-70M / WikiText-2 (surgery-only). k=128 is full context; smaller k use exact tail.
PYTHIA_SMOKE_KS := 32 128
PYTHIA_KS := 1 2 4 8 16 32 64 128
PYTHIA_SMOKE_TARGETS := $(foreach k,$(PYTHIA_SMOKE_KS),artifacts/checkpoints/ts_surgery_pythia_70m_topk$(k)_smoke_tailmass_exact.pt)
PYTHIA_SURGERY_TARGETS := $(foreach k,$(PYTHIA_KS),artifacts/checkpoints/ts_surgery_pythia_70m_topk$(k)_fast_tailmass_exact.pt)
PYTHIA_TAILMASS0_TARGETS := $(foreach k,$(PYTHIA_KS),artifacts/checkpoints/ts_surgery_pythia_70m_topk$(k)_fast_tailmass0.pt)

# Query-centroid support routing: fixed k-means query centroids (calibrated on train data) select
# the sparse support; exact scores are read only on it. Calibrated / exact / tail-drop tail modes.
ROUTING_KS := 8 16 32 64
ROUTING_TARGETS := $(foreach k,$(ROUTING_KS),$(foreach m,_qc32 _qc32_tailmass_exact _qc32_tailmass0,artifacts/checkpoints/ts_surgery_topk$(k)_fast$(m).pt)) \
	artifacts/checkpoints/ts_surgery_topk16_fast_qc8.pt artifacts/checkpoints/ts_surgery_topk16_fast_qc128.pt
PYTHIA_ROUTING_KS := 8 16 32
PYTHIA_ROUTING_TARGETS := $(foreach k,$(PYTHIA_ROUTING_KS),$(foreach m,_qc32_tailmass_exact _qc32_tailmass0,artifacts/checkpoints/ts_surgery_pythia_70m_topk$(k)_fast$(m).pt)) \
	artifacts/checkpoints/ts_surgery_pythia_70m_topk16_fast_qc8_tailmass_exact.pt \
	artifacts/checkpoints/ts_surgery_pythia_70m_topk16_fast_qc128_tailmass_exact.pt

STRICT_ROUTING_TARGETS := $(foreach k,$(STRICT_KS),artifacts/checkpoints/ts_surgery_topk$(k)_strict_qc32.pt)

# Strict graph without exp and log (squares, sigmoid, inverse square root) and with the
# default strict products (strict_product=norm: norm expansion where a square is shared, polarization elsewhere; the
# polarization-only runs are in artifacts/backup_polarization_2026-10-10/),
# with exact GELU (_sr) or GELU ~ x*sigmoid(1.702x) (_sr_gs), routed strict variants, and PTQ of the
# new strict graphs. The extended graph keeps exact GELU: NPUs with the extended basis implement it.
SR_KS := 16 32 64 197
STRICT_SR_TARGETS := $(foreach k,$(SR_KS),$(foreach m,_strict_sr _strict_sr_gs,artifacts/checkpoints/ts_surgery_topk$(k)$(m).pt)) \
	$(foreach k,$(STRICT_KS),$(foreach m,_strict_sr_qc32 _strict_sr_gs_qc32,artifacts/checkpoints/ts_surgery_topk$(k)$(m).pt))
PTQ_STRICT_SR_TARGETS := $(foreach k,$(SR_KS),$(foreach m,strict_sr strict_sr_pertensor strict_sr_gs strict_sr_gs_pertensor,artifacts/checkpoints/ts_ptq_$(k)_$(m)_8bit.pt))

# Hadamard graph (hadamard=true): the strict graph with every product of two activations, squares included,
# as one elementwise product instead of polarization, and QK^T and PV as products plus sums (no matmul).
# Unary maps: sigmoid and inverse square root, plus exact GELU (_hadamard_sr) or none (_hadamard_sr_gs);
# _qc32 is the routed support of 32 query centroids.
HADAMARD_SR_TARGETS := $(foreach k,$(SR_KS),$(foreach m,_hadamard_sr _hadamard_sr_gs,artifacts/checkpoints/ts_surgery_topk$(k)$(m).pt)) \
	$(foreach k,$(STRICT_KS),$(foreach m,_hadamard_sr_qc32 _hadamard_sr_gs_qc32,artifacts/checkpoints/ts_surgery_topk$(k)$(m).pt))

all: a b c d e a_imagenet mambair

# A. Main Fast-Path Top-k Results
a: $(SURGERY_MAIN_TARGETS) $(DISTILL_MAIN_CE_TARGETS) $(DISTILL_MAIN_J_TARGETS)

# A2. ImageNet Fast-Path Top-k Results
a_imagenet: $(IMAGENET_SURGERY_MAIN_TARGETS)

# B. Tail-Mass Top-k Ablation
b: $(TAIL_SURGERY_FIXED0_TARGETS) $(TAIL_SURGERY_FIXED1E3_TARGETS) $(TAIL_SURGERY_CAL_TARGETS) $(TAIL_DISTILL_TARGETS)

# C. Strict vs Fast Representability/Efficiency Experiment
c: $(STRICT_FAST_TARGETS) $(STRICT_STRICT_TARGETS)

# D. PTQ Experiments
d: $(PTQ_TARGETS)

# E. Exact Tail-Mass Fast-Path Top-k Sweep (Pet)
e: $(TAIL_SURGERY_EXACT_TARGETS)

# MambaIRv2 Light SR: surgery + distill sweep (x2 and x4)
mambair: $(MAMBAIR_SURGERY_TARGETS) $(MAMBAIR_DISTILL_TARGETS)
mambair-x2: $(MAMBAIR_X2_TARGETS)
mambair-x4: $(MAMBAIR_X4_TARGETS)
# Exact runtime q_tail (use_exact_tail_mass) surgery + distill sweep for x4.
mambair-x4-exact: $(MAMBAIR_X4_EXACT_TARGETS)
# Tail-drop (q=0) surgery + distill sweep for x4.
mambair-x4-tailmass0: $(MAMBAIR_X4_TAIL0_TARGETS)

# Download DIV2K train + Set5 benchmark into ./data (loaders also auto-download on demand).
mambair-data:
	$(PYTHON) -m transformer_surgery.cli.download_sr --root ./data --scale 2
	$(PYTHON) -m transformer_surgery.cli.download_sr --root ./data --scale 4

# Pythia-70M WikiText-2 surgery-only (requires: pip install -e '.[llm]').
pythia-smoke: $(PYTHIA_SMOKE_TARGETS)
pythia: $(PYTHIA_SURGERY_TARGETS)
pythia-tailmass0: $(PYTHIA_TAILMASS0_TARGETS)

# Query-centroid support routing (surgery-only; routed checkpoints are not reloaded downstream).
routing: $(ROUTING_TARGETS)
pythia-routing: $(PYTHIA_ROUTING_TARGETS)
# Strict graph with routed support (scores formed only on the support), and a same-session cost bench
# against strict top-k (accuracy, latency, peak memory, unary evaluations per image).
routing-strict: $(STRICT_ROUTING_TARGETS)
strict-sr: $(STRICT_SR_TARGETS)
hadamard-sr: $(HADAMARD_SR_TARGETS)

# Per-module CUDA-event profile of the Hadamard graph with GELU ~ x*sigmoid(1.702x) (k=16,32,64), with the
# strict and extended k=16 checkpoints as same-session controls. The exact-GELU Hadamard profile of the same
# day is kept in artifacts/logs/profile_hadamard_2026-10-10/.
PROFILE_HADAMARD_LOG := artifacts/logs/profile_hadamard_gs_2026-10-10
PROFILE_HADAMARD_CKPTS := $(foreach k,$(STRICT_KS),artifacts/checkpoints/ts_surgery_topk$(k)_hadamard_sr_gs.pt) \
	artifacts/checkpoints/ts_surgery_topk16_strict_sr.pt artifacts/checkpoints/ts_surgery_topk16_fast.pt
# Paper Table S4 in one profiling session: strict and Hadamard graphs with GELU ~ x*sigmoid(1.702x), top-k and
# routed, and the extended graph with exact GELU (k=16,32,64 for the strict and extended shares).
PROFILE_GS_LOG := artifacts/logs/profile_gs_2026-10-10
PROFILE_GS_CKPTS := $(foreach k,$(STRICT_KS),artifacts/checkpoints/ts_surgery_topk$(k)_strict_sr_gs.pt) \
	artifacts/checkpoints/ts_surgery_topk16_strict_sr_gs_qc32.pt artifacts/checkpoints/ts_surgery_topk16_hadamard_sr_gs.pt \
	artifacts/checkpoints/ts_surgery_topk16_hadamard_sr_gs_qc32.pt \
	$(foreach k,$(STRICT_KS),artifacts/checkpoints/ts_surgery_topk$(k)_fast.pt)
profile-gs: $(PROFILE_GS_CKPTS)
	mkdir -p $(PROFILE_GS_LOG)
	echo "$(PYTHON) scripts/profile_modules.py --checkpoints $^ --out $(PROFILE_GS_LOG)/profile.json" > $(PROFILE_GS_LOG)/command.txt
	$(PYTHON) scripts/profile_modules.py --checkpoints $^ --out $(PROFILE_GS_LOG)/profile.json | tee $(PROFILE_GS_LOG)/profile.log

# Polarization baseline of Table 2: the strict graph with polarization for every product (strict_product=polarization,
# per-term mix), GELU ~ x*sigmoid(1.702x), top-k and routed support.
STRICT_POL_TARGETS := $(foreach k,$(SR_KS),artifacts/checkpoints/ts_surgery_topk$(k)_strict_sr_gs_pol.pt) \
	$(foreach k,$(STRICT_KS),artifacts/checkpoints/ts_surgery_topk$(k)_strict_sr_gs_pol_qc32.pt)
strict-pol: $(STRICT_POL_TARGETS)

# bf16 error of the strict value mix on real inputs: polarization, norm expansion with partial sums, the default
# per-term construction, and the per-term construction with a balanced operand scale (paper Supplementary).
MIX_PRECISION_LOG := artifacts/logs/mix_precision_2026-10-10
mix-precision: artifacts/checkpoints/ts_surgery_topk16_strict_sr.pt artifacts/checkpoints/ts_surgery_topk197_strict_sr.pt
	mkdir -p $(MIX_PRECISION_LOG)
	$(PYTHON) scripts/mix_precision.py 16 197 | tee $(MIX_PRECISION_LOG)/mix_precision.log

# bf16 against fp32 at full support (k = N) for every network: the reference and each compiled graph (paper
# Supplementary). Reads the existing k = N surgery checkpoints; no prerequisites, so it never rebuilds them.
DTYPE_CHECK_LOG := artifacts/logs/dtype_check_2026-10-17
dtype-check:
	mkdir -p $(DTYPE_CHECK_LOG)
	$(PYTHON) scripts/dtype_check.py $(DTYPE_CHECK_LOG)/dtype_check.json | tee $(DTYPE_CHECK_LOG)/dtype_check.log

# Node-level check of Brent's operand scaling under PTQ-style integer quantization (8/10/12-bit input codes of the
# compiled affine nodes, max-abs grids, exact accumulation): Score, Mix and LayerNorm of the strict graph (paper Supplementary S8).
PTQ_BALANCE_LOG := artifacts/logs/ptq_balance_2026-10-11
ptq-balance-check: artifacts/checkpoints/ts_surgery_topk16_strict_sr_gs.pt artifacts/checkpoints/ts_surgery_topk197_strict_sr_gs.pt
	mkdir -p $(PTQ_BALANCE_LOG)
	$(PYTHON) scripts/ptq_balance_check.py | tee $(PTQ_BALANCE_LOG)/ptq_balance_check.log

profile-hadamard: $(PROFILE_HADAMARD_CKPTS)
	mkdir -p $(PROFILE_HADAMARD_LOG)
	echo "$(PYTHON) scripts/profile_modules.py --checkpoints $^ --out $(PROFILE_HADAMARD_LOG)/profile.json" > $(PROFILE_HADAMARD_LOG)/command.txt
	$(PYTHON) scripts/profile_modules.py --checkpoints $^ --out $(PROFILE_HADAMARD_LOG)/profile.json | tee $(PROFILE_HADAMARD_LOG)/profile.log
ptq-strict-sr: $(PTQ_STRICT_SR_TARGETS)
routing-strict-bench: $(foreach k,$(STRICT_KS),artifacts/checkpoints/ts_surgery_topk$(k)_strict_sr.pt artifacts/checkpoints/ts_surgery_topk$(k)_strict_sr_qc32.pt) $(STRICT_FAST_TARGETS) $(foreach k,$(STRICT_KS),artifacts/checkpoints/ts_surgery_topk$(k)_fast_qc32.pt)
	$(PYTHON) scripts/bench_strict_routing.py strict_sr $(STRICT_KS)
	$(PYTHON) scripts/bench_strict_routing.py fast $(STRICT_KS)

# Float64 end-to-end check of exact compilation (all keys, extended and strict graphs vs the reference).
EXACT_FP64_LOG := artifacts/logs/exact_fp64_2026-10-10
exact-fp64: artifacts/checkpoints/ts_surgery_topk197_fast.pt artifacts/checkpoints/ts_surgery_topk197_strict_sr.pt
	mkdir -p $(EXACT_FP64_LOG)
	$(PYTHON) scripts/check_exact_fp64.py --checkpoints $^ --batches 2 | tee $(EXACT_FP64_LOG)/check.log

# Same float64 check for the Hadamard graph (own log, so the exact-fp64 log is kept).
EXACT_FP64_HADAMARD_LOG := artifacts/logs/exact_fp64_hadamard_2026-10-10
exact-fp64-hadamard: artifacts/checkpoints/ts_surgery_topk197_hadamard_sr.pt
	mkdir -p $(EXACT_FP64_HADAMARD_LOG)
	$(PYTHON) scripts/check_exact_fp64.py --checkpoints $^ --batches 2 | tee $(EXACT_FP64_HADAMARD_LOG)/check.log

help:
	@echo "Targets:"
	@echo "  make a      # Main fast-path top-k sweep"
	@echo "  make b      # Tail-mass ablation"
	@echo "  make c      # Strict (polarization products, exact GELU) vs fast surgery"
	@echo "  make d      # PTQ granularity ablation (fast and polarization strict graphs)"
	@echo "  make e      # Exact runtime tail-mass pet-fast sweep"
	@echo "  make a_imagenet  # ImageNet fast-path surgery sweep"
	@echo "  make mambair-data # Download DIV2K + Set5 (x2 and x4 LR) into ./data"
	@echo "  make mambair     # MambaIR Light SR surgery+distill (x2 and x4)"
	@echo "  make mambair-x2  # MambaIR x2 sweep only"
	@echo "  make mambair-x4  # MambaIR x4 sweep only"
	@echo "  make mambair-x4-exact  # MambaIR x4 exact-tail surgery+distill (all k)"
	@echo "  make mambair-x4-tailmass0  # MambaIR x4 tail-drop (q=0) surgery+distill (k=1..64)"
	@echo "  make pythia-smoke # Pythia-70M capped WikiText-2 smoke (topk32 + topk128 exact-tail)"
	@echo "  make pythia       # Pythia-70M full exact-tail surgery sweep (k=1..128)"
	@echo "  make pythia-tailmass0 # Pythia-70M zero fixed-tail (no exact mass) sweep (k=1..128)"
	@echo "  make routing      # DeiT Pet query-centroid support routing (k=8..64, calibrated/exact/tail-drop)"
	@echo "  make pythia-routing # Pythia-70M query-centroid support routing (k=8..32, exact/tail-drop)"
	@echo "  make routing-strict # polarization strict graph with routed support"
	@echo "  make routing-strict-bench # same-session cost: strict_sr and fast, top-k vs routed"
	@echo "  make exact-fp64   # float64 end-to-end check of exact compilation (k=197)"
	@echo "  make strict-sr    # strict graph without exp/log (+GELU approx) and routed strict"
	@echo "  make ptq-strict-sr # 8-bit PTQ of the new strict graphs"
	@echo "  make dtype-check   # bf16 vs fp32 at k = N, all networks (reference and compiled graphs)"
	@echo "  make hadamard-sr  # Hadamard graph: elementwise products instead of polarization (sigmoid, rsqrt)"
	@echo "  make exact-fp64-hadamard # float64 check of the Hadamard graph (k=197)"
	@echo "  make profile-hadamard # per-module GPU profile of the Hadamard graph (k=16,32,64)"
	@echo "  make profile-gs   # Table S4 profile: strict/Hadamard with GELU approx, routed strict, extended"
	@echo "  make mix-precision # bf16 error of the strict value mix: polarization vs squares (paper Supplementary)"
	@echo "  make strict-pol   # Table 2 polarization baseline (strict_product=polarization), top-k and routed"
	@echo "  make all    # Run all sections"

# Surgery checkpoints rebuild when surgery JSON changes.
artifacts/checkpoints/ts_surgery_%.pt: configs/surgery/%.json
	$(PYTHON) -m transformer_surgery.cli.surgery --config "$<"

# ImageNet surgery checkpoints rebuild when ImageNet surgery JSON changes.
$(sort $(IMAGENET_SURGERY_TARGETS)): artifacts/checkpoints/ts_surgery_%_imagenet.pt: configs/surgery_imagenet/%_imagenet.json
	$(PYTHON) -m transformer_surgery.cli.surgery --config "$<"

# Distill checkpoints rebuild when distill JSON or prerequisite surgery checkpoint changes.
define DISTILL_RULE
artifacts/checkpoints/ts_distill_$(1)_fast_$(2).pt: configs/distill/$(1)_fast_$(2).json artifacts/checkpoints/ts_surgery_topk$(1)_fast.pt
	$$(PYTHON) -m transformer_surgery.cli.distill --config "$$<"
endef

$(foreach k,$(MAIN_KS),$(eval $(call DISTILL_RULE,$(k),ce)))
$(foreach k,$(MAIN_KS),$(eval $(call DISTILL_RULE,$(k),jeffreys)))

# Optional strict distillation config (kept explicit since naming differs).
artifacts/checkpoints/ts_distill_64_strict_jeffreys.pt: configs/distill/64_strict_jeffreys.json artifacts/checkpoints/ts_surgery_topk64_strict.pt
	$(PYTHON) -m transformer_surgery.cli.distill --config "$<"

# MambaIR surgery checkpoints reuse the generic `ts_surgery_%.pt: configs/surgery/%.json` rule.
# MambaIR distill rebuilds when its JSON or the prerequisite surgery checkpoint changes.
define MAMBAIR_DISTILL_RULE
artifacts/checkpoints/ts_distill_mambair_x$(1)_topk$(2).pt: configs/distill/mambair_x$(1)_topk$(2).json artifacts/checkpoints/ts_surgery_mambair_x$(1)_topk$(2)_fast.pt
	$$(PYTHON) -m transformer_surgery.cli.distill --config "$$<"
endef

$(foreach s,$(MAMBAIR_SCALES),$(foreach k,$(MAMBAIR_KS),$(eval $(call MAMBAIR_DISTILL_RULE,$(s),$(k)))))

# Exact-tail MambaIR distill depends on the matching exact-tail surgery checkpoint.
define MAMBAIR_EXACT_DISTILL_RULE
artifacts/checkpoints/ts_distill_mambair_x$(1)_topk$(2)_tailmass_exact.pt: configs/distill/mambair_x$(1)_topk$(2)_tailmass_exact.json artifacts/checkpoints/ts_surgery_mambair_x$(1)_topk$(2)_fast_tailmass_exact.pt
	$$(PYTHON) -m transformer_surgery.cli.distill --config "$$<"
endef

$(foreach k,$(MAMBAIR_KS),$(eval $(call MAMBAIR_EXACT_DISTILL_RULE,4,$(k))))

# Tail-drop MambaIR distill depends on the matching tail-drop surgery checkpoint.
define MAMBAIR_TAIL0_DISTILL_RULE
artifacts/checkpoints/ts_distill_mambair_x$(1)_topk$(2)_tailmass0.pt: configs/distill/mambair_x$(1)_topk$(2)_tailmass0.json artifacts/checkpoints/ts_surgery_mambair_x$(1)_topk$(2)_fast_tailmass0.pt
	$$(PYTHON) -m transformer_surgery.cli.distill --config "$$<"
endef

$(foreach k,$(MAMBAIR_TAIL0_KS),$(eval $(call MAMBAIR_TAIL0_DISTILL_RULE,4,$(k))))

# PTQ checkpoints rebuild when PTQ JSON or source float checkpoint changes.
define PTQ_RULE
artifacts/checkpoints/ts_ptq_$(1).pt: configs/ptq/$(1).json artifacts/checkpoints/$(2).pt
	$$(PYTHON) -m transformer_surgery.cli.ptq --config "$$<"
endef

$(foreach k,$(PTQ_KS),$(eval $(call PTQ_RULE,$(k)_fast_jeffreys_8bit,ts_distill_$(k)_fast_jeffreys)))
$(foreach k,$(PTQ_KS),$(eval $(call PTQ_RULE,$(k)_fast_jeffreys_pertensor_8bit,ts_distill_$(k)_fast_jeffreys)))
$(foreach k,$(PTQ_KS),$(eval $(call PTQ_RULE,$(k)_strict_8bit,ts_surgery_topk$(k)_strict)))
$(foreach k,$(PTQ_KS),$(eval $(call PTQ_RULE,$(k)_strict_pertensor_8bit,ts_surgery_topk$(k)_strict)))
$(foreach k,$(SR_KS),$(eval $(call PTQ_RULE,$(k)_strict_sr_8bit,ts_surgery_topk$(k)_strict_sr)))
$(foreach k,$(SR_KS),$(eval $(call PTQ_RULE,$(k)_strict_sr_pertensor_8bit,ts_surgery_topk$(k)_strict_sr)))
$(foreach k,$(SR_KS),$(eval $(call PTQ_RULE,$(k)_strict_sr_gs_8bit,ts_surgery_topk$(k)_strict_sr_gs)))
$(foreach k,$(SR_KS),$(eval $(call PTQ_RULE,$(k)_strict_sr_gs_pertensor_8bit,ts_surgery_topk$(k)_strict_sr_gs)))
# Brent scaling at model level (seeded PTQ calibration): strict graph with GELU ~ x*sigmoid(1.702x), k = 16, 64, 197,
# 12- or 10-bit input codes on the compiled fixed affine nodes (compiled_affine_input_bits; "acc12"/"acc10" in the
# config names), without and with brent_scaling, per-channel first, then per-tensor.
PTQ_BRENT_NAMES := \
	16_strict_sr_gs_acc12_8bit 16_strict_sr_gs_acc12_brent_8bit 16_strict_sr_gs_acc10_8bit 16_strict_sr_gs_acc10_brent_8bit \
	64_strict_sr_gs_acc12_8bit 64_strict_sr_gs_acc12_brent_8bit 64_strict_sr_gs_acc10_8bit 64_strict_sr_gs_acc10_brent_8bit \
	197_strict_sr_gs_acc12_8bit 197_strict_sr_gs_acc12_brent_8bit 197_strict_sr_gs_acc10_8bit 197_strict_sr_gs_acc10_brent_8bit \
	16_strict_sr_gs_acc12_pertensor_8bit 16_strict_sr_gs_acc12_brent_pertensor_8bit 16_strict_sr_gs_acc10_pertensor_8bit 16_strict_sr_gs_acc10_brent_pertensor_8bit \
	64_strict_sr_gs_acc12_pertensor_8bit 64_strict_sr_gs_acc12_brent_pertensor_8bit 64_strict_sr_gs_acc10_pertensor_8bit 64_strict_sr_gs_acc10_brent_pertensor_8bit \
	197_strict_sr_gs_acc12_pertensor_8bit 197_strict_sr_gs_acc12_brent_pertensor_8bit 197_strict_sr_gs_acc10_pertensor_8bit 197_strict_sr_gs_acc10_brent_pertensor_8bit
$(foreach n,$(PTQ_BRENT_NAMES),$(eval $(call PTQ_RULE,$(n),ts_surgery_topk$(firstword $(subst _, ,$(n)))_strict_sr_gs)))
ptq-brent: $(foreach n,$(PTQ_BRENT_NAMES),artifacts/checkpoints/ts_ptq_$(n).pt)
# Same with lam on the value mix only (brent_sites=["mix"]): folding lam into the LayerNorm costs bf16 accuracy.
PTQ_BRENT_MIX_NAMES := \
	16_strict_sr_gs_acc12_brentmix_8bit 16_strict_sr_gs_acc10_brentmix_8bit 64_strict_sr_gs_acc12_brentmix_8bit 64_strict_sr_gs_acc10_brentmix_8bit \
	197_strict_sr_gs_acc12_brentmix_8bit 197_strict_sr_gs_acc10_brentmix_8bit 16_strict_sr_gs_acc12_brentmix_pertensor_8bit 16_strict_sr_gs_acc10_brentmix_pertensor_8bit \
	64_strict_sr_gs_acc12_brentmix_pertensor_8bit 64_strict_sr_gs_acc10_brentmix_pertensor_8bit 197_strict_sr_gs_acc12_brentmix_pertensor_8bit 197_strict_sr_gs_acc10_brentmix_pertensor_8bit
$(foreach n,$(PTQ_BRENT_MIX_NAMES),$(eval $(call PTQ_RULE,$(n),ts_surgery_topk$(firstword $(subst _, ,$(n)))_strict_sr_gs)))
ptq-brent-mix: $(foreach n,$(PTQ_BRENT_MIX_NAMES),artifacts/checkpoints/ts_ptq_$(n).pt)
