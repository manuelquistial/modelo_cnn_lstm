# Das et al. (2025) EEG Motor Imagery Replication Pipeline

[![Repository](https://img.shields.io/badge/GitHub-manuelquistial%2Fmodelo__cnn__lstm-blue)](https://github.com/manuelquistial/modelo_cnn_lstm)

Research-grade replication of **"Enhanced EEG signal classification in brain computer interfaces using hybrid deep learning models"** (Das et al., 2025) for a master thesis on EEG-based motor imagery BCI.

**Remote:** `git@github.com:manuelquistial/modelo_cnn_lstm.git`  
**Runtime:** TensorFlow/Keras — optimizado para **Paperspace GPU** → ver [PAPERSPACE.md](PAPERSPACE.md).

## Features

- PhysioNet EEG Motor Movement/Imagery Dataset (103 subjects after exclusions)
- Binary (left/right) and five-class motor imagery modes
- ROI-based channel selection (6 ROIs)
- Band-pass filtering (0.5–50 Hz), configurable epoch lengths (1s, 4s, 5s)
- Traditional ML: KNN, SVM, Logistic Regression, Random Forest, Naive Bayes
- Deep learning: CNN, LSTM, CNN-LSTM-Attention (main hybrid model)
- Optional WGAN-GP data augmentation (train only)
- Riemannian geometry baselines (pyriemann)
- Subject-wise and trial-wise splits (no leakage in subject-wise mode)
- PCA/t-SNE visualizations, confusion matrices, ROC curves
- CSV/JSON outputs and error analysis by subject/trial

## Installation

### Local

```bash
git clone git@github.com:manuelquistial/modelo_cnn_lstm.git
cd modelo_cnn_lstm
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Paperspace (GPU)

```bash
git clone git@github.com:manuelquistial/modelo_cnn_lstm.git
cd modelo_cnn_lstm
./scripts/paperspace_setup.sh
QUICK=1 ./scripts/paperspace_run.sh
```

Detalle completo: **[PAPERSPACE.md](PAPERSPACE.md)**.

**Optional dependencies**

| Package | Purpose |
|---------|---------|
| `pyriemann` | Riemannian classifiers (skipped if missing) |
| `PyWavelets` | Wavelet features (skipped if missing) |
| `tensorflow` | Deep models and GAN (required for DL) |

On first run, MNE downloads PhysioNet data automatically (~several GB for all subjects).

## Quick start (smoke test)

```bash
python -m das2025_replication.run_experiments --quick --no-roi --no-segment
```

Uses 5 subjects, skips ROI loop and segment-length sweep.

## Full experiment

```bash
python -m das2025_replication.run_experiments
```

Or:

```bash
python run_pipeline.py --max-subjects 20 --epochs 50
```

## Python API

```python
from das2025_replication.config import get_paper_subjects
from das2025_replication.data_loading import build_dataset, print_dataset_summary
from das2025_replication.config import BINARY_CLASS_NAMES
from das2025_replication.run_experiments import run_complete_das2025_replication

subjects = get_paper_subjects()  # 103 IDs
X, y, meta, ch = build_dataset(subjects[:10], mode="binary", roi_name="ROI_6")
print_dataset_summary(X, y, meta, BINARY_CLASS_NAMES)

results = run_complete_das2025_replication(
    mode="binary",
    split_strategy="subjectwise",
    max_subjects=10,
    dl_epochs=25,
)
```

## Project structure

```
das2025_replication/
  config.py           # Constants, ROIs, subjects
  data_loading.py     # PhysioNet load, epoching, event mapping
  rois.py             # ROI channel selection
  preprocessing.py    # Splits, normalization, CSP, ICA
  features.py         # Time/freq/wavelet features for ML
  models_ml.py        # sklearn classifiers
  models_deep.py      # CNN, LSTM, CNN-LSTM-Attention
  gan.py              # Conditional WGAN-GP augmentation
  riemannian.py       # pyriemann baselines
  evaluation.py       # Metrics, predictions, error analysis
  visualization.py    # Plots (CM, ROC, t-SNE, etc.)
  run_experiments.py  # Full orchestration
```

## Outputs

Saved under `outputs/das2025_replication/`:

| File | Description |
|------|-------------|
| `dataset_summary.csv` | Trial/subject counts |
| `ml_metrics.csv` | Traditional ML results |
| `deep_metrics.csv` | CNN/LSTM/hybrid results |
| `roi_results.csv` | Per-ROI comparison |
| `segment_length_results.csv` | 1s/4s/5s × 25/50 epochs |
| `riemannian_results.csv` | Riemannian baselines |
| `predictions_by_trial.csv` | Per-trial predictions |
| `errors_by_subject.csv` | Subject-level errors |
| `misclassified_trials.csv` | Misclassified trials |
| `experiment_config.json` | Run configuration |
| `figures/` | All plots |

## Fidelity to Das et al. (2025)

See **[docs/PAPER_AUDIT_Das2025.md](docs/PAPER_AUDIT_Das2025.md)** for a line-by-line comparison with the article.

To run closer to the paper's reported setup (640×2 contralateral input, trial-wise split):

```bash
python -m das2025_replication.run_experiments \
  --mode multiclass \
  --trialwise \
  --paper-input \
  --paper-roi-epochs \
  --epochs 50
```

## Methodological notes

1. **Subject-wise split** (`split_strategy="subjectwise"`) is recommended for thesis work — no subject appears in both train and test.
2. **Trial-wise split** reproduces a paper-like setup but may inflate accuracy via subject leakage.
3. Results are **not** tuned to match the paper's reported ~96% accuracy; honest cross-subject performance is expected to be lower.
4. The paper mentions both five-class and binary tasks; both are implemented.
5. Segment length: primary = 5s (800 samples @ 160 Hz); 4s (640 samples) included for the article's 640×2 inconsistency.
6. All scalers, CSP, PCA (when used for ML), and GANs are fit **only on training data**.

## Assumptions

- PhysioNet event codes T0=1, T1=2, T2=3 after `events_from_annotations`
- ROI channel names match standard 10–20 labels in the dataset
- ICA artifact removal is optional and off by default
- GAN training is expensive; disabled by default (`--gan` to enable)

## License

For academic/thesis use. PhysioNet data subject to its own terms.
