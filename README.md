# WGAN-GP augmentation for minority intrusion classes

This is a small experiment with CICIDS2017. It checks whether a classifier gets better at recognizing minority attacks when its training set includes samples from WGAN-GP models.

In these runs, the answer was “not consistently.” Bot recall improved, but the change in macro F1 was smaller than the variation between runs. The generated XSS rows were also much less diverse than the real ones.

## Experiment

Three versions of the same MLP classifier are compared:

1. real training data only;
2. real data plus WGAN-GP samples;
3. real data with balanced sample weights.

Bot, Web Attack - XSS, and DoS Slowhttptest each have their own generator and critic. These are class-specific WGAN-GPs, not a conditional WGAN-GP: neither network receives a class label.

Generated rows are added to training data only. All three classifiers use the same real validation and test sets.

## Dataset

The input is the flow CSV release of CICIDS2017. This pilot keeps seven labels:

- BENIGN
- DDoS
- DoS Hulk
- PortScan
- Bot
- Web Attack - XSS
- DoS Slowhttptest

Classes represented by only a few dozen rows are not included. A GAN trained on so little data would be difficult to assess, and the corresponding test metrics would be unstable.

The loader uses deterministic reservoir sampling, capped at 20,000 rows per class. It selected 88,117 rows for this run. Cleaning removed 127 invalid rows and 7,004 exact duplicates. Three feature vectors occurred with conflicting labels, so all 12 rows containing those vectors were removed.

| Split | Rows |
|---|---:|
| Train | 56,681 |
| Validation | 12,146 |
| Test | 12,147 |

The variance filter, scaler, and PCA are fitted once, using the real training split. PCA kept 23 components, covering at least 95% of the variance in that split. WGAN-GP training and sampling happen in the resulting PCA space.

## Model details

For one class, $P_r$ denotes the real training distribution, $P_g$ the generated distribution, $G$ the generator, and $D$ the critic. The critic minimizes:

$$
L_D =
\mathbb{E}_{\tilde{x} \sim P_g}[D(\tilde{x})]
- \mathbb{E}_{x \sim P_r}[D(x)]
+ \lambda \mathbb{E}_{\hat{x} \sim P_{\hat{x}}}
\left(\lVert \nabla_{\hat{x}}D(\hat{x}) \rVert_2 - 1\right)^2
$$

The penalty is evaluated at points between real and generated rows:

$$
\hat{x} = \epsilon x + (1 - \epsilon)\tilde{x},
\qquad \epsilon \sim U(0,1)
$$

The generator minimizes:

$$
L_G = -\mathbb{E}_{z \sim U([-1,1]^d)}[D(G(z))]
$$

Here, $x$ is real, $\tilde{x}=G(z)$ is generated, $z$ is a $d$-dimensional noise vector, and $\lambda=10$ is the gradient-penalty coefficient.

Both networks have hidden layers of 128, 128, and 64 units. A generator runs for 400 steps; every generator step follows five critic updates. Adam uses a learning rate of $10^{-4}$. Synthetic rows bring each target class to 5,000 training rows.

The weighted condition uses the inverse-frequency weight:

$$
w_c = \frac{N}{C n_c}
$$

Here, $N$ is the real training-set size, $C$ the number of classes, and $n_c$ the training count for class $c$.

## Metrics

Macro F1 is the main metric. Each run also saves per-class precision, recall and F1, a confusion matrix, and one-vs-rest ROC-AUC. Minority-class false-negative rate is calculated as:

$$
\mathrm{FNR}_c =
\frac{\mathrm{FN}_c}{\mathrm{TP}_c + \mathrm{FN}_c}
= 1 - \mathrm{Recall}_c
$$

Every class has both positive and negative test examples, so ROC-AUC is defined here. Seeds 42, 1, and 2 keep the data split fixed while changing initialization and stochastic training.

## Results

<!-- GENERATED_RESULTS_START -->

| Condition | Macro F1, mean ± SD | Macro ROC-AUC, mean ± SD | Runs |
|---|---:|---:|---:|
| Baseline | 0.9721 ± 0.0038 | 0.99909 ± 0.00004 | 3 |
| WGAN-GP augmented | 0.9729 ± 0.0052 | 0.99890 ± 0.00006 | 3 |
| Weighted loss | 0.9613 ± 0.0056 | 0.99902 ± 0.00013 | 3 |

<!-- GENERATED_RESULTS_END -->

Augmentation changed macro F1 by $0.0008 \pm 0.0070$ relative to the matching baseline runs. Some seeds improved and others did not, so there is no stable overall gain in this pilot.

The class-level changes are more informative:

- Bot recall increased from 0.8824 to 0.9155.
- XSS recall stayed at 0.9898, while its F1 decreased from 0.9327 to 0.9282.
- Slowhttptest recall stayed at 0.9957.

![Macro F1 by condition](figures/macro_f1_conditions.png)

![Minority-class false-negative rates](figures/minority_fnr.png)

The confusion matrices below are row-normalized and averaged over three seeds. Both panels come from the same real test rows.

![Baseline and augmented confusion matrices](figures/confusion_matrices.png)

## Synthetic sample check

Each seed generated 9,520 rows: 3,636 Bot, 4,544 XSS, and 1,340 Slowhttptest. A basic diversity check compares mean pairwise distance in the synthetic sample with the same measurement in the corresponding real training sample.

The three-seed mean synthetic-to-real ratios are:

- Bot: 0.740
- DoS Slowhttptest: 0.622
- Web Attack - XSS: 0.145

XSS is the clear problem: its generated rows occupy a much narrower region than the real sample. This is consistent with mode collapse or strong under-dispersion. Bot and Slowhttptest look better by this measure, although pairwise distance alone cannot validate either distribution.

![Synthetic diversity diagnostic](figures/synthetic_diversity.png)

## Reproducing the run

Python 3.11–3.14 is supported. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,gan,report]"
```

Put one merged CICIDS2017 CSV in `data/raw/`, or pass another path with `--csv`. Use either the merged CSV or the eight daily files, since passing both duplicates the flows.

Prepare the fixed split:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py `
  --config configs\pilot.toml `
  --csv data\raw\CICIDS2017.csv `
  --output data\processed\pilot
```

Run one seed:

```powershell
$seed = 42

.\.venv\Scripts\python.exe scripts\run_baseline.py `
  --config configs\pilot.toml --prepared data\processed\pilot `
  --output "results\pilot\baseline\seed_$seed" --seed $seed

.\.venv\Scripts\python.exe scripts\train_wgan_gp.py `
  --config configs\pilot.toml --prepared data\processed\pilot `
  --output "results\pilot\wgan\seed_$seed" --seed $seed

.\.venv\Scripts\python.exe scripts\run_condition.py `
  --condition augmented --config configs\pilot.toml `
  --prepared data\processed\pilot `
  --synthetic "results\pilot\wgan\seed_$seed\synthetic_training_only.npz" `
  --output "results\pilot\augmented\seed_$seed" --seed $seed

.\.venv\Scripts\python.exe scripts\run_condition.py `
  --condition weighted --config configs\pilot.toml `
  --prepared data\processed\pilot `
  --output "results\pilot\weighted\seed_$seed" --seed $seed
```

Repeat those commands for seeds 1 and 2, then rebuild the aggregate files and figures:

```powershell
.\.venv\Scripts\python.exe scripts\aggregate_metrics.py `
  --results-root results\pilot --seeds 42 1 2 `
  --output results\pilot\aggregate.json

.\.venv\Scripts\python.exe scripts\generate_report_assets.py `
  --results-root results\pilot --figures figures `
  --readme README.md --seeds 42 1 2

.\.venv\Scripts\python.exe scripts\check_reproducibility.py `
  --prepared data\processed\pilot `
  --results-root results\pilot --seeds 42 1 2

.\.venv\Scripts\python.exe -m pytest
```

Preparation records the input filename and SHA-256 checksum. Model runs save metrics, predictions, losses, runtime, package versions, and platform details. Git ignores raw data and large model artifacts; compact JSON metrics, CSV summaries, and figures remain available.

## Limitations

This result should be read as a pilot rather than a benchmark. The dataset is capped, and rows are split randomly at flow level. Similar flows from one capture session may land in different splits, which could explain the high absolute scores. Three seeds expose some initialization variance but do not support strong statistical claims.

PCA preserves high-variance directions, not necessarily the features that best separate attacks. The diversity check says nothing about privacy and does not prove that a generated row is novel. XSS synthetic rows outnumber real XSS training rows by roughly ten to one, making their low diversity particularly concerning.

The next useful run would use a day- or session-aware split, lower synthetic-to-real ratios, validation-based generator stopping, and direct comparisons with random oversampling and SMOTE. Precision-recall AUC, calibration, and nearest-neighbor privacy checks are also missing from this pilot.
