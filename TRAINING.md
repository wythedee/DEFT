**Main datasets**

| Dataset | Task | Input shape | #Cls | Train / Val / Test |
|---|---|---:|---:|---|
| BCIC-IV-2a | Motor imagery | `(22,4,200)` | 4 | 5 / 2 / 2 subjects; 2784 / 1152 / 1152 samples |
| FACED | Emotion recognition | `(32,10,200)` | 9 | 80 / 20 / 23 subjects; 6720 / 1680 / 1932 samples |
| ISRUC-S1 | Sleep staging | `(20,6,6000)` | 5 | 80 / 10 / 10 subjects; 3559 / 468 / 435 samples |
| MentalArith. | Workload assessment | `(20,5,200)` | 2 | 26 / 5 / 5 subjects; 1247 / 240 / 220 samples |
| BCIC-Speech | Speech decoding | `(60,3,200)` | 3 | 9 / 3 / 3 subjects; 4050 / 1350 / 1350 samples |
| Mumtaz2016 | MDD classification | `(19,5,200)` | 2 | 43 / 9 / 10 subjects; 4891 / 1041 / 1151 samples |

Common setting: subject-disjoint split; most datasets are resampled to 200 Hz and organized as `(channels, windows, 200)`; EEG values are divided by 100 in the loaders; test data are used only for final reporting.

**Backbones**

| Backbone | Setting |
|---|---|
| CodeBrain | released checkpoint; downstream fine-tuning |
| CBraMod | released checkpoint; downstream fine-tuning |
| LaBraM | released checkpoint; downstream fine-tuning |
| CSBrain | released checkpoint; downstream fine-tuning |

No EFM pretraining is rerun. The pretrained backbone, downstream head, and disentanglement adapter are fine-tuned during downstream training.

**Common hyperparameters**

| Hyperparameter | Value / policy |
|---|---|
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `0.05` |
| Dropout | `0.1` |
| Label smoothing | `0.1` |
| Task loss weight | `1.0` |
| Random seed | main table multi-seed: `0--2`; appendix seed-0 |
| Model selection | best checkpoint selected on validation split |
| Hardware | NVIDIA RTX 4090 |
| Adapter hidden dim | `200` |
| Gaussian transition components | `16` |
| Remove-gate initialization | `-2.0` |
| Contrastive temperature | `0.07` |
| Variance-ratio target | `2.0` |

**Backbone-specific hyperparameters**

| Backbone | LR | Batch size | Epochs | `lambda_ord` | `lambda_sty` | `lambda_con` | `lambda_rec` |
|---|---:|---:|---:|---:|---:|---:|---:|
| CodeBrain | `1e-4` | 64 | 20 | 0.005 | 0.001 | 0.02 | 0.01 |
| CBraMod | `1e-4` | 64 | 20 | 0.1 | 0.01--0.02 | 0.2 | 0.02 |
| LaBraM | `1e-4` | 64 | 20 | 0.1 | 0.02 | 0.2 | 0.02 |
| CSBrain | `1e-4` | 64 | 20 | 0.1 | 0.01 | 0.01 | 0.02 |

