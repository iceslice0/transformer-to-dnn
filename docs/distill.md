# Distillation

Distillation is the optional fine-tuning stage after surgery. It starts from a surgery checkpoint,
loads the adapter reference checkpoint as teacher, and trains the surgery student with a mixed
hard-label CE plus Jeffreys teacher-matching objective.

The core code lives in ``src/transformer_surgery/distill.py``. The CLI wrapper is
``python -m transformer_surgery.cli.distill`` or ``ts-distill``.

## Inputs

- ``pre_checkpoint``: surgery checkpoint to fine-tune.
- ``reference_checkpoint``: teacher/reference checkpoint, resolved through the model adapter.
- ``model_key``: adapter key, default ``deit_tiny_pet``.
- dataset/config fields shared with the adapter, such as ``data_dir``, ``batch_size``, and ``workers``.
- optimization fields: ``epochs``, ``lr``, ``weight_decay``, ``warmup_epochs``, ``grad_clip``,
  ``cosine_eta_min``, ``keep_best``, and progress intervals.

Distillation must not import Pet/DeiT code directly. It uses ``get_model_adapter`` and
``load_surgery_student_checkpoint`` so model-specific loading, dataloaders, and reconstruction stay
inside the adapter layer.

## Objective

For each batch:

- teacher logits are computed under ``torch.no_grad()``.
- student logits are trained with hard-label cross entropy.
- teacher/student distribution mismatch is measured with ``jeffreys_divergence_dense``.
- the loss is ``(1 - distill_weight) * CE + distill_weight * Jeffreys``.

``temperature`` is passed to the Jeffreys metric. Validation reports student accuracy, mean CE, and
mean Jeffreys divergence against the teacher.

## Training Rule

Trainable:

- affine weights and biases where ``requires_grad`` is set.
- PWL parameters, including trainable knot values where present.
- ``gamma``/``beta`` in ``AffineScaleBias``.

Not trainable:

- fixed ``eps`` buffers/parameters frozen by ``freeze_eps_parameters``.
- fixed coefficients and scalar constants registered by the surgery graph.
- teacher parameters.

When the runtime device is CUDA and the surgery dtype is ``float16``, distillation uses a float32
training copy and copies trainable state back into the surgery student. Other dtypes train the
student directly.

## Outputs

The CLI writes:

- ``artifacts/checkpoints/ts_distill_<config>.pt``.
- ``artifacts/metadata/ts_distill_<config>.json``.

Checkpoint metadata records the adapter key, teacher checkpoint, source surgery checkpoint,
temperature, distillation weight, validation metrics, best epoch, config JSON path, and surgery
dtype. The metadata JSON adds post-distillation metrics under the ``calibration`` block so the full
surgery -> distill result remains traceable by checkpoint basename.

## Acceptance Criteria

The distillation stage is acceptable only if:

1. It loads the student through the adapter checkpoint path and does not depend on a concrete model module.
2. It keeps the teacher frozen.
3. It reports CE and Jeffreys validation metrics before saving.
4. It saves a real PyTorch checkpoint and matching metadata derived from the output checkpoint name.
5. It preserves fixed surgery constants and only trains intended parameters.
