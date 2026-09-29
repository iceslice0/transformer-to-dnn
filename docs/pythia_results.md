# Pythia-70M Surgery Results

WikiText-2 validation, fixed non-overlapping windows of `context_length=128`,
`surgery_dtype=float32`, `allow_matmul=true`.
Reference: Hugging Face `EleutherAI/pythia-70m` (eager attention).
Metric: token NLL (lower is better) and perplexity \(e^{\mathrm{NLL}}\).

Evaluated on **252,544** tokens.

- **exact**: `use_exact_tail_mass=true` (runtime omitted-tail mass).
- **tail0**: `gibbs_tail_prob_eps=0`, `use_exact_tail_mass=false` (dropped mass not reinjected).
- **eps**: `gibbs_tail_prob_eps_exact_{mean,std,min,max}` from surgery metadata
  (dense-softmax omitted-tail mass on causal masked scores over calibration batches).

## Full top-k sweep

| top_k | exact NLL | Δ NLL | exact PPL | Δ PPL | tail0 NLL | Δ NLL | tail0 PPL | Δ PPL | eps mean±std (min, max) |
|------:|----------:|------:|----------:|------:|----------:|------:|----------:|------:|------------------------:|
| ref | 4.5496 | — | 94.59 | — | 4.5496 | — | 94.59 | — | — |
| 128 | 4.5496 | -0.0000 | 94.59 | -0.00 | 4.5496 | -0.0000 | 94.59 | -0.00 | 0.0000±0.0000 (0.0000, 0.0000) |
| 64 | 4.5497 | +0.0002 | 94.61 | +0.01 | 4.5497 | +0.0002 | 94.61 | +0.02 | 0.0041±0.0156 (0.0000, 0.2311) |
| 32 | 4.5513 | +0.0017 | 94.75 | +0.16 | 4.5567 | +0.0072 | 95.27 | +0.68 | 0.0232±0.0579 (0.0000, 0.4760) |
| 16 | 4.5584 | +0.0088 | 95.43 | +0.84 | 4.5885 | +0.0390 | 98.35 | +3.76 | 0.0616±0.1151 (0.0000, 0.6969) |
| 8 | 4.5822 | +0.0326 | 97.73 | +3.14 | 4.6811 | +0.1316 | 107.89 | +13.30 | 0.1232±0.1796 (0.0000, 0.8187) |
| 4 | 4.6534 | +0.1038 | 104.94 | +10.35 | 4.9232 | +0.3737 | 137.44 | +42.85 | 0.2046±0.2411 (0.0000, 0.8961) |
| 2 | 4.8267 | +0.2771 | 124.80 | +30.21 | 5.5418 | +0.9923 | 255.14 | +160.55 | 0.3135±0.2955 (0.0000, 0.9471) |
| 1 | 5.1804 | +0.6308 | 177.76 | +83.16 | 6.6381 | +2.0885 | 763.65 | +669.06 | 0.4405±0.3269 (0.0000, 0.9741) |

Configs: `configs/surgery/pythia_70m_topk{K}_fast_tailmass_exact.json` and
`configs/surgery/pythia_70m_topk{K}_fast_tailmass0.json`.
