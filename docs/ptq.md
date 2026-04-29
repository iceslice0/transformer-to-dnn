LEAN CUT-PASTE AGENT PLAN FOR STANDALONE PTQ SCRIPT

Goal

Create one standalone Python + PyTorch script that performs PTQ on a selected subset of affine nodes in an already surged fp model.

The script should:

1. load the fp model and checkpoint
2. attach hooks to collect calibration data from selected affine/matmul nodes
3. run validation on the fp model while collecting calibration data
4. build a wrapped model by replacing selected affine/matmul nodes with quantize -> integer affine/matmul -> dequantize wrappers
5. initialize wrapper parameters from collected calibration data
6. validate the wrapped model

No training.
No QAT.
Only PTQ initialization and evaluation.

Use python -m transformer_surgery.cli.run_surgery as reference, reuse code if possible

File to create

python -m transformer_surgery.cli.ptq


Inputs

JSON Config file with
1. path to fp checkpoint
2. node selection flags (Linear/Conv, Affine, MatMul)
3. explicit node include/exclude lists by name
5. calibration subset size
6. quantization config:
   - weight bits = 8
   - activation bits = 8
   - ``per_output_channel``: for ``Linear``/``Conv2d`` only — **True** = per-output-channel weight scales, **False** = one global weight scale (usually very bad accuracy). Affine/unary/coeff always use one global weight scale; ``MatMul`` has no weights. Output **scale** for ``Linear``/``Conv2d`` is ``s_in * s_w``; only **bias** is calibrated. **Input** activations: one scale per input tensor (tensor-wide max).


Outputs

1. wrapped model checkpoint
2. wrapper metadata json
3. printed validation metrics:
   - fp model metrics
   - wrapped model metrics
4. optional debug stats per wrapped node (mean, std for inputs and outputs)

Scope

Wrap only selected affine/matmul nodes.
Leave all other nodes untouched.

Affine/Matmul nodes means explicit modules of the form:
- Linear, Conv
- Affine
- MatMul

High-level flow

Step 1
Load fp model definition and checkpoint.

Step 2
Select affine/matmul nodes to calibrate/wrap.

Step 3
Attach forward hooks to selected affine nodes in the fp model.
For each wrapped candidate collect statistics on:
- input activation x (x, w for matmul)
- output activation y_fp

Step 4
Run validation on the fp model with hooks enabled.
Use validation subset of required size

At the end you should have cached calibration tensors/statistics for each selected affine/matmul node.

Step 5
Create wrapped model by deep-copying the fp model.

Step 6
Replace selected affine/matmul nodes in the copied model with PTQ wrappers.

Step 7
Initialize each wrapper from:
- original float weights/bias
- collected calibration input/output data

Step 8
Run validation on the wrapped model.

Step 9
Save wrapped model checkpoint and metadata.


Required components

1. Node selection helper
Given model.named_modules(), select affine/matmul nodes by:
- module type
- name patterns
- explicit include/exclude lists


2. Calibration hooks

For each selected affine node attach a forward hook that stores:
- x_in = module input tensor
- w_in = module weight tensor (for MatMul)
- y_out = module output tensor

Keep storage bounded by using channelwise running moments min/max and/or histograms

Minimum needed per node:
- sample of input tensor values
- sample of output tensor values
- float weight tensor
- float bias tensor if present

3. Quantization wrapper

Create one common wrapper class, e.g.

CalibratedAffinePTQWrapper

The wrapper should contain:
- original affine structure
- quantized weight representation or weight quantizer for MatMul
- input quantizer params
- output dequant params

Forward logic should be conceptually:

x_fp, w_fp, x_zeropoint_fp, w_zeropoint_fp, x_scale_fp, w_scale_fp, out_scale_fp, out_zeropoint_fp
-> quantize input to int_n proxy
-> quantize weight to int_n proxy
-> integer affine accumulator
-> dequantize out by fp affine
-> return out fp tensor

Important:
the implementation semantics should be:
- input quantized
- weight quantized
- accumulator integer
- output dequantized back to fp

Absorb bias into output dequant affine map.


4. Wrapper parameters to initialize

For each wrapped affine node initialize:

A. Weight and Input quantization scale, zeropoint

Choose from float weights using either:
- min, max
- mean, std, sigma_scale (param, e.g. 3*sigma, 6*sigma)


B. Output dequant parameters
Fit affine dequantization from integer accumulator output to teacher fp output.

Use either:
- sampled moment matching (account for clamps/rounding in quantized input and weights)
- least-squares affine fit


Recommended output model:

y_hat = s_out * y_int + c_out

Fit per output channel:
- s_out
- c_out

using cached teacher outputs.

This is the key calibration step.


5. Initialization procedure per wrapped node

For each selected affine node:

1. read float weights and bias
2. compute weight quant scale
3. quantize weights
4. read cached input calibration tensor
5. compute input activation scale
6. quantize cached input
7. run simulated integer affine on cached input
8. fit output dequant parameters to teacher output:
   - either by moment matching
   - or by least-squares affine fit

Preferred:
- least-squares affine fit per output channel


6. Validation passes

Run two validations:

A. Float validation
- original model
- hooks enabled
- collect calibration data
- log baseline metrics

B. Wrapped validation
- wrapped model
- no hooks needed unless debugging
- log wrapped metrics

Report:
- original accuracy/loss
- wrapped accuracy/loss
- difference


7. Save artifacts

Save:
- wrapped checkpoint
- metadata json

Metadata should include:
- selected wrapped node names
- calibration batch count
- quantization config
- per-node scales
- per-node dequant params
- validation metrics before/after wrapping


Implementation details to keep it lean

1. Do not quantize Not Affine/MatMul nodes (Unary, selection/routing).
Do not quantize selection/routing ops.
Only wrap selected affine nodes.

2. Do not do global optimization.
Per-node one-pass calibration only.

3. Do not do iterative refinement.
Single pass only.

4. Do not require integer-only output chaining.
Returning dequantized fp output from wrappers is acceptable for the PTQ baseline.


Suggested internal structure of ptq_wrap_validate.py

1. parse args
2. load model
3. load checkpoint
4. build dataloaders
5. select nodes
6. register hooks
7. validate fp model and collect calibration data
8. remove hooks
9. deepcopy model
10. replace selected nodes with wrappers and initialize wrappers from calibration data
11. validate wrapped model
12. save checkpoint and metadata


Acceptance criteria

The script is acceptable if:

1. it runs end-to-end from checkpoint to wrapped checkpoint
2. it can wrap only a selected subset of affine nodes
3. it collects calibration data from the fp teacher model in one pass
4. it initializes wrapper params from calibration data
5. it validates both original and wrapped models
6. it saves wrapped checkpoint and metadata


Final instruction to agent

Implement the minimal standalone PTQ script.

Focus on:
- one-pass teacher calibration
- selected affine node wrapping
- per-node scale fitting
- output affine dequant fitting
- before/after validation

Do not add:
- QAT
- vendor runtime
- full-graph quantization
- unary and selection/routing quantization