.PHONY: all a b c d e a_imagenet mambair mambair-x2 mambair-x4 mambair-x4-exact mambair-x4-tailmass0 mambair-data pythia-smoke pythia pythia-tailmass0 help

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

# Main experiment sweeps.
MAIN_KS := 1 2 4 8 16 32 64 96 128 197
TAIL_KS := 1 2 4 8 16 32 64
STRICT_KS := 16 32 64
PTQ_KS := 16 32 64 197
IMAGENET_MAIN_KS := 1 4 32 64 197

# MambaIRv2 Light SR sweep. k=0 is uniform attention (affine key-mean, not Gibbs);
# k=256 is full window support (window_size=16 -> seq_len=256).
MAMBAIR_KS := 0 1 2 4 16 64
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

help:
	@echo "Targets:"
	@echo "  make a      # Main fast-path top-k sweep"
	@echo "  make b      # Tail-mass ablation"
	@echo "  make c      # Strict vs fast surgery"
	@echo "  make d      # PTQ granularity ablation"
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
