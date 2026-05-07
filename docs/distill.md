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
  ``cosine_eta_min``, ``keep_best``, and ``train_progress_interval``.
- repeat fields: ``base_seed`` and ``num_trainings``. Training run ``i`` uses seed
  ``base_seed + i``.

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

The optimizer is built over all student parameters (``train_student.parameters()``); it is not
filtered by ``requires_grad``. Effective trainability is governed by ``requires_grad`` flags and
buffer-vs-parameter status:

Trainable:

- ``nn.Linear`` and ``nn.Conv2d`` weights and biases.
- ``AffineScaleBias.weight``/``bias`` (LayerNorm ``gamma``/``beta``).
- ``GibbsTopKSoftmax.gibbs_tail_prob_eps`` (an ``nn.Parameter``, not frozen).

Not trainable:

- ``AffineContract`` and ``AffineFixedMix`` coefficients - registered as buffers (``coeff``,
  ``weight``), so they never enter ``parameters()``.
- fixed ``eps`` floors in ``NLLogPlusEps``/``NLRsqrtPlusEps``/``NLReciprocalPlusEps`` - registered
  as buffers.
- teacher parameters (``requires_grad`` is set to ``False`` before training).

When the runtime device is CUDA and the surgery dtype is ``float16``, distillation uses a float32
training copy and copies its full state dict back into the surgery student after each epoch. Other
dtypes train the student directly.

When ``num_trainings`` is greater than one, each run reloads the same ``pre_checkpoint`` and trains
independently. All run metrics are kept in metadata, validation accuracy mean/std are computed
across runs, and only the best-validation-accuracy checkpoint is written to disk.

## Outputs

The CLI writes:

- ``artifacts/checkpoints/ts_distill_<config>.pt``.
- ``artifacts/metadata/ts_distill_<config>.json``.

Checkpoint metadata records the adapter key, teacher checkpoint, source surgery checkpoint,
temperature, distillation weight, base seed, number of trainings, every run's metrics, validation
accuracy mean/std, best run, config JSON path, and surgery dtype. The saved model state is from the
best run only. The metadata JSON adds post-distillation metrics under the ``calibration`` block so
the full surgery -> distill result remains traceable by checkpoint basename.

## Validation expectations

The distillation stage is is expected to satisfy:

1. It loads the student through the adapter checkpoint path and does not depend on a concrete model module.
2. It keeps the teacher frozen.
3. It reports CE and Jeffreys validation metrics before saving.
4. It keeps per-run metrics, reports validation accuracy mean/std, and saves only the best run's
   checkpoint.
5. It saves a real PyTorch checkpoint and matching metadata derived from the output checkpoint name.
6. It preserves fixed surgery constants (eps buffers, registered coefficient buffers) and only
   trains parameters that have ``requires_grad=True``.
