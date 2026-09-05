# Code-Mixed Text Classification Model Evaluation

A lightweight AI/ML project for training and evaluating sentiment classification models on **code-mixed Nepali-English text**. The repository includes preprocessing, model training, multi-model benchmarking, and result artifact generation.

## Why this project?

Code-mixed social text (e.g., Nepali + English in one sentence) is noisy, informal, and difficult for standard monolingual NLP systems. This project focuses on evaluating how well different multilingual and Nepal-aware sentiment models perform under code-mixing conditions.

## What this project evaluates

- End-to-end preprocessing for code-mixed sentiment data
- Fine-tuning of a transformer-based classifier
- Comparative evaluation against multiple baseline sentiment models
- Breakdown of performance by estimated code-mixing level
- Output artifacts: per-sentence predictions, summary metrics, and plots

## Dataset overview

The workflow expects TSV data with columns such as:

- `id`
- `text`
- `sentiment` (3-class sentiment labels)

If your dataset schema differs, adapt the preprocessing/training scripts accordingly.

## Evaluation workflow

1. **Preprocess** text (normalization, cleanup, label mapping) using `preprocess.py`
2. **Train/Fine-tune** the core sentiment model using `train.py`
3. **Evaluate** across multiple models and generate comparison artifacts using `evaluate_final.py`
4. **Inspect** generated CSVs and figures for model-level and mixing-level performance

## Metrics used

The repository computes classification metrics including:

- Accuracy
- Macro F1-score
- Precision (macro)
- Recall (macro)
- Confusion matrix (best model)
- Accuracy by estimated code-mixing level (`low`, `medium`, `high`)

## Repository structure

```text
.
├── preprocess.py                         # Text normalization + split loading
├── train.py                              # Fine-tuning script
├── evaluate_final.py                     # Multi-model evaluation + plots
├── demo.py                               # Interactive inference demo
├── summary_metrics.csv                   # Overall model comparison output
├── accuracy_by_mixing_level.csv          # Performance by mixing level
├── predictions.csv                       # Per-sentence predictions
├── fig1_overall_accuracy.png             # Accuracy bar chart
├── fig2_accuracy_by_mixing_level.png     # Mixing-level comparison chart
├── fig3_confusion_matrix_best_model.png  # Best-model confusion matrix
└── nepali-english-sentiment-project.zip  # Dataset/project artifact bundle
```

## Setup / installation

Use Python 3.9+ (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers datasets scikit-learn pandas numpy matplotlib nltk
```

> If your environment, CUDA setup, or package versions differ, adapt dependency versions as needed.

## How to run

### 1) Training

```bash
python train.py \
  --train data/train.tsv \
  --val data/val.tsv \
  --model_name xlm-roberta-base \
  --epochs 4 \
  --batch_size 16 \
  --output_dir outputs/xlmr-sentiment
```

### 2) Evaluation / model comparison

```bash
# Update file paths/model path inside evaluate_final.py if needed
python evaluate_final.py
```

### 3) Demo inference

```bash
python demo.py --model_dir final
```

> **Note:** Paths such as `data/train.tsv`, `data/val.tsv`, `test.tsv`, and model directories are environment-specific placeholders unless those files already exist in your local setup.

## Results / findings

Use this template to document your latest benchmark run:

| Model | Accuracy | Macro F1 | Notes |
|---|---:|---:|---|
| Fine-tuned model | _TBD_ | _TBD_ | |
| Baseline A | _TBD_ | _TBD_ | |
| Baseline B | _TBD_ | _TBD_ | |
| Baseline C | _TBD_ | _TBD_ | |

Generated artifacts to support analysis:

- `summary_metrics.csv`
- `accuracy_by_mixing_level.csv`
- `predictions.csv`
- `fig1_overall_accuracy.png`, `fig2_accuracy_by_mixing_level.png`, `fig3_confusion_matrix_best_model.png`

## Limitations and future work

- Mixing-level estimation currently uses a heuristic and may misclassify language proportion.
- Robustness across dialects, scripts, and domains should be tested further.
- Add experiment tracking, reproducible configs, and fixed dependency versions.
- Add a formal test suite and CI for reliability.

## Contributing

Contributions are welcome. Suggested process:

1. Fork the repository
2. Create a feature branch
3. Make focused changes with clear commit messages
4. Open a pull request describing motivation, approach, and results

## License

No license file is currently present in this repository. Until clarified by the owner, treat this project as **unlicensed / all rights reserved**.

## Contact / author

Repository owner: **[@ProgAbhishek](https://github.com/ProgAbhishek)**

For questions or collaboration, please open an issue in this repository.
