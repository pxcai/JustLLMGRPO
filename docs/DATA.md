# Data preparation

## Required annotation columns

The training and validation CSV files must include:

| Column | Description |
|---|---|
| `annotated_prompt` | Source CXR generation prompt derived from a radiology report. |
| `chexpert_labels` | Python-literal or dictionary mapping of the 14 CheXpert labels to `1`, `0`, or `-1`. |
| `id` | Stable sample identifier used to construct matched deterministic seeds. |

Optional columns retained in the VERL ground-truth payload are:

| Column | Description |
|---|---|
| `path` | Relative real-image path. |
| `view` | View metadata. |
| `orientation` | AP/PA or related orientation metadata. |

The expected CheXpert label names are:

```text
Atelectasis
Cardiomegaly
Consolidation
Edema
Enlarged Cardiomediastinum
Fracture
Lung Lesion
Lung Opacity
No Finding
Pleural Effusion
Pleural Other
Pneumonia
Pneumothorax
Support Devices
```

## VERL parquet conversion

The training launcher automatically converts CSV files to `train.parquet` and `val.parquet`. To run the conversion directly:

```bash
python -m llm_sana.data.prepare_llavarad_prompt_parquet \
  --train_csv /path/to/train.csv \
  --val_csv /path/to/test.csv \
  --output_dir data/processed/llavarad_prompt_rewrite \
  --balanced_val_per_label 40 \
  --seed 42
```

The paper protocol selects 40 positive validation rows for each of the 14 labels. Set `--balanced_val_per_label 0` for a general validation CSV or smoke test.

## Controlled-access data

MIMIC-CXR and its images are not distributed in this repository. Users must complete the required credentialing and data-use agreement through PhysioNet. Keep all patient-derived data outside version control. The provided `.gitignore` excludes common medical-image and checkpoint formats.
