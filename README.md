# Project 1 — Emotion Classification

## Task

Given a short text (e.g. a social-media post), classify it into one of **7 emotion categories**:

| Label | Emotion  |
|-------|----------|
| 0     | anger    |
| 1     | disgust  |
| 2     | fear     |
| 3     | joy      |
| 4     | neutral  |
| 5     | sadness  |
| 6     | surprise |

The primary evaluation metric is **Macro-F1**.

## Data

All data files are located in `data/`:

| File               | Description                              |
|--------------------|------------------------------------------|
| `train.csv`        | Training set (columns: `id`, `text`, `label`) |
| `valid.csv`        | Validation set (same format as train)    |
| `test_no_label.csv`| Test set — labels are withheld (columns: `id`, `text`) |

## Dependencies

```bash
pip install torch pandas scikit-learn
```

## Baselines

All baseline scripts must be **run from the `project_1/` directory**.

### MLP (Embedding + Mean-Pool + 3-layer MLP)

```bash
python baselines/mlp.py
```

Saves predictions to `emb_mlp_pred.csv`.

### Bi-RNN

```bash
python baselines/rnn.py
```

Saves predictions to `rnn_pred.csv`.

## Evaluation

To evaluate predictions on the **validation set**, run:

```bash
python evaluate.py --pred <path_to_pred.csv>
```

Example:

```bash
python evaluate.py --pred emb_mlp_pred.csv
```

The script prints accuracy, macro-precision, macro-recall, and macro-F1, along with a per-class breakdown.

> **Note:** The `evaluate.py` script evaluates against `data/valid.csv`. For the final test set, submit your `prediction.csv` (columns: `id`, `label`) following the course submission instructions.

## Prediction Format

Your prediction file must be a CSV with exactly two columns: `id` and `label`.

```
id,label
eebbqej,4
ed00q6i,4
...
```
