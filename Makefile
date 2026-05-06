.PHONY: all a b c d a_imagenet help

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

# Main experiment sweeps.
MAIN_KS := 1 2 4 8 16 32 64 96 128 192
TAIL_KS := 1 2 4 8 16 32 64
STRICT_KS := 16 32 64
PTQ_KS := 16 32 64
IMAGENET_MAIN_KS := 1 4 32 64 192

SURGERY_MAIN_TARGETS := $(foreach k,$(MAIN_KS),artifacts/checkpoints/ts_surgery_topk$(k)_fast.pt)
IMAGENET_SURGERY_MAIN_TARGETS := $(foreach k,$(IMAGENET_MAIN_KS),artifacts/checkpoints/ts_surgery_topk$(k)_fast_imagenet.pt)
DISTILL_MAIN_CE_TARGETS := $(foreach k,$(MAIN_KS),artifacts/checkpoints/ts_distill_$(k)_fast_ce.pt)
DISTILL_MAIN_J_TARGETS := $(foreach k,$(MAIN_KS),artifacts/checkpoints/ts_distill_$(k)_fast_jeffreys.pt)

TAIL_SURGERY_FIXED0_TARGETS := $(foreach k,$(TAIL_KS),artifacts/checkpoints/ts_surgery_topk$(k)_fast_tailmass0.pt)
TAIL_SURGERY_FIXED1E3_TARGETS := $(foreach k,$(TAIL_KS),artifacts/checkpoints/ts_surgery_topk$(k)_fast_tailmass1e3.pt)
TAIL_SURGERY_CAL_TARGETS := $(foreach k,$(TAIL_KS),artifacts/checkpoints/ts_surgery_topk$(k)_fast.pt)
TAIL_DISTILL_TARGETS := $(foreach k,$(TAIL_KS),artifacts/checkpoints/ts_distill_$(k)_fast_jeffreys.pt)

STRICT_FAST_TARGETS := $(foreach k,$(STRICT_KS),artifacts/checkpoints/ts_surgery_topk$(k)_fast.pt)
STRICT_STRICT_TARGETS := $(foreach k,$(STRICT_KS),artifacts/checkpoints/ts_surgery_topk$(k)_strict.pt)
IMAGENET_SURGERY_TARGETS := $(IMAGENET_SURGERY_MAIN_TARGETS)

PTQ_FAST_CHANNEL_TARGETS := $(foreach k,$(PTQ_KS),artifacts/checkpoints/ts_ptq_$(k)_fast_jeffreys_8bit.pt)
PTQ_FAST_TENSOR_TARGETS := $(foreach k,$(PTQ_KS),artifacts/checkpoints/ts_ptq_$(k)_fast_jeffreys_pertensor_8bit.pt)
PTQ_STRICT_CHANNEL_TARGETS := $(foreach k,$(PTQ_KS),artifacts/checkpoints/ts_ptq_$(k)_strict_8bit.pt)
PTQ_STRICT_TENSOR_TARGETS := $(foreach k,$(PTQ_KS),artifacts/checkpoints/ts_ptq_$(k)_strict_pertensor_8bit.pt)
PTQ_TARGETS := $(PTQ_FAST_CHANNEL_TARGETS) $(PTQ_FAST_TENSOR_TARGETS) $(PTQ_STRICT_CHANNEL_TARGETS) $(PTQ_STRICT_TENSOR_TARGETS)

all: a b c d a_imagenet

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

help:
	@echo "Targets:"
	@echo "  make a      # Main fast-path top-k sweep"
	@echo "  make b      # Tail-mass ablation"
	@echo "  make c      # Strict vs fast surgery"
	@echo "  make d      # PTQ granularity ablation"
	@echo "  make a_imagenet  # ImageNet fast-path surgery sweep"
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

# PTQ checkpoints rebuild when PTQ JSON or source float checkpoint changes.
define PTQ_RULE
artifacts/checkpoints/ts_ptq_$(1).pt: configs/ptq/$(1).json artifacts/checkpoints/$(2).pt
	$$(PYTHON) -m transformer_surgery.cli.ptq --config "$$<"
endef

$(foreach k,$(PTQ_KS),$(eval $(call PTQ_RULE,$(k)_fast_jeffreys_8bit,ts_distill_$(k)_fast_jeffreys)))
$(foreach k,$(PTQ_KS),$(eval $(call PTQ_RULE,$(k)_fast_jeffreys_pertensor_8bit,ts_distill_$(k)_fast_jeffreys)))
$(foreach k,$(PTQ_KS),$(eval $(call PTQ_RULE,$(k)_strict_8bit,ts_surgery_topk$(k)_strict)))
$(foreach k,$(PTQ_KS),$(eval $(call PTQ_RULE,$(k)_strict_pertensor_8bit,ts_surgery_topk$(k)_strict)))
