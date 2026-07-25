# Exact Tail-Mass vs Calibrated (Pet Fast)

Pre-finetune val accuracy and omitted-tail mass after surgery (`make a` calibrated fast vs `make e` exact).
Accuracies are `student_pre_ft_val_acc`. Calibrated mean \(q\) is `gibbs_tail_prob_eps_calibrated_mean`.
Exact \(q\) stats are pooled over train calibration rows (`gibbs_tail_prob_eps_exact_*`).

| k | calib acc | exact acc | calib mean q | exact mean q ± std (min, max) |
|---:|---:|---:|---:|---|
| 1 | 14.94% | 7.88% | 0.7823 | 0.7825 ± 0.1658 (0.000401, 0.9937) |
| 2 | 39.36% | 19.90% | 0.7230 | 0.7137 ± 0.1794 (0.0002192, 0.9877) |
| 4 | 64.73% | 56.83% | 0.6349 | 0.6292 ± 0.1899 (0.0002283, 0.9745) |
| 8 | 79.29% | 81.38% | 0.5302 | 0.5276 ± 0.1923 (8.035e-05, 0.9503) |
| 16 | 85.45% | 88.83% | 0.4125 | 0.4118 ± 0.1799 (7.153e-06, 0.9025) |
| 32 | 88.85% | 90.90% | 0.2832 | 0.2829 ± 0.1493 (0, 0.8071) |
| 64 | 90.84% | 91.44% | 0.1508 | 0.1511 ± 0.0983 (0, 0.6411) |
| 96 | 91.09% | 91.69% | 0.0812 | 0.0811 ± 0.0606 (0, 0.4729) |
| 128 | 91.31% | 91.61% | 0.0398 | 0.0394 ± 0.0327 (0, 0.3097) |
| 197 | 91.63% | 91.63% | 0.0000 | 0.0000 ± 0.0000 (0, 0.0000) |
