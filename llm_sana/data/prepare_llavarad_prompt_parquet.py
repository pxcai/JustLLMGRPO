#!/usr/bin/env python3
"""Convert LLaVA-Rad CheXGenBench annotations to verl parquet format."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llm_sana.prompts import build_messages


CHEXPERT_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


def _parse_labels(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    return ast.literal_eval(str(value))


def _is_positive(value: Any) -> bool:
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def _build_rows(df: pd.DataFrame, split: str, prompt_col: str, labels_col: str) -> list[dict[str, Any]]:
    rows = []
    for idx, row in df.iterrows():
        original_prompt = str(row[prompt_col]).strip()
        labels = _parse_labels(row[labels_col]) if labels_col in row else {}
        row_id = row["id"] if "id" in row else idx
        view = row["view"] if "view" in row else ""
        orientation = row["orientation"] if "orientation" in row else ""
        balanced_val_label = row.get("balanced_val_label", None)
        extra_info = {
            "split": split,
            "index": int(idx),
            "row_id": row_id,
            "original_prompt": original_prompt,
            "chexpert_labels": labels,
            "view": view if not pd.isna(view) else "",
            "orientation": orientation if not pd.isna(orientation) else "",
            "source_image_path": row["path"] if "path" in row else "",
        }
        if balanced_val_label is not None and not pd.isna(balanced_val_label):
            extra_info["balanced_val_label"] = str(balanced_val_label)
        rows.append(
            {
                "data_source": "chexgenbench/llavarad_prompt_rewrite",
                "prompt": build_messages(original_prompt),
                "ability": "cxr_prompt_planning",
                "reward_model": {
                    "style": "chexgenbench_v3",
                    "ground_truth": {
                        "original_prompt": original_prompt,
                        "chexpert_labels": labels,
                        "row_id": row_id,
                        "view": view if not pd.isna(view) else "",
                        "orientation": orientation if not pd.isna(orientation) else "",
                        "source_image_path": row["path"] if "path" in row else "",
                        **(
                            {"balanced_val_label": str(balanced_val_label)}
                            if balanced_val_label is not None and not pd.isna(balanced_val_label)
                            else {}
                        ),
                    },
                },
                "extra_info": extra_info,
            }
        )
    return rows


def _load_csv(
    path: Path,
    prompt_col: str,
    labels_col: str,
    max_samples: int,
    seed: int,
    deduplicate_prompts: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in (prompt_col, labels_col):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in {path}")

    df = df.dropna(subset=[prompt_col]).copy()
    df[prompt_col] = df[prompt_col].astype(str).str.strip()
    df = df[df[prompt_col] != ""]
    if deduplicate_prompts:
        df = df.drop_duplicates(subset=[prompt_col])
    df = df.reset_index(drop=True)
    if max_samples and max_samples > 0:
        df = df.sample(n=min(max_samples, len(df)), random_state=seed).reset_index(drop=True)
    return df


def _select_balanced_by_label(
    df: pd.DataFrame,
    labels_col: str,
    labels: list[str],
    per_label: int,
    seed: int,
) -> pd.DataFrame:
    if per_label <= 0:
        return df

    parsed_labels = df[labels_col].map(_parse_labels)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    selected_set: set[int] = set()
    selected_label_by_idx: dict[int, str] = {}

    # Sample rare labels first so the validation set stays disjoint where possible.
    label_counts = {
        label: int(parsed_labels.map(lambda item, label=label: _is_positive(item.get(label))).sum()) for label in labels
    }
    for label in sorted(labels, key=lambda item: label_counts[item]):
        candidates = [idx for idx, item in parsed_labels.items() if _is_positive(item.get(label))]
        if len(candidates) < per_label:
            raise ValueError(f"Label {label!r} only has {len(candidates)} positive samples, need {per_label}.")
        candidates = rng.permutation(candidates).tolist()
        chosen = [idx for idx in candidates if idx not in selected_set][:per_label]
        if len(chosen) < per_label:
            raise ValueError(
                f"Could not select {per_label} disjoint validation samples for label {label!r}; "
                f"got {len(chosen)} after earlier labels."
            )
        for idx in chosen:
            selected.append(idx)
            selected_set.add(idx)
            selected_label_by_idx[idx] = label

    out = df.loc[selected].copy().reset_index(drop=True)
    out["balanced_val_label"] = [selected_label_by_idx[idx] for idx in selected]
    return out


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    df.to_parquet(tmp_path)
    tmp_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt_col", default="annotated_prompt")
    parser.add_argument("--labels_col", default="chexpert_labels")
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--balanced_val_per_label", type=int, default=0)
    parser.add_argument("--balanced_val_labels", default=",".join(CHEXPERT_LABELS))
    parser.add_argument("--deduplicate_prompts", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [label.strip() for label in args.balanced_val_labels.split(",") if label.strip()]
    train_df = _load_csv(
        Path(args.train_csv),
        args.prompt_col,
        args.labels_col,
        args.max_train_samples,
        args.seed,
        deduplicate_prompts=args.deduplicate_prompts,
    )
    if args.val_csv:
        val_df = _load_csv(
            Path(args.val_csv),
            args.prompt_col,
            args.labels_col,
            0 if args.balanced_val_per_label > 0 else args.max_val_samples,
            args.seed,
            deduplicate_prompts=args.deduplicate_prompts,
        )
        if args.balanced_val_per_label > 0:
            val_df = _select_balanced_by_label(
                val_df,
                labels_col=args.labels_col,
                labels=labels,
                per_label=args.balanced_val_per_label,
                seed=args.seed,
            )
    else:
        val_size = max(1, int(round(len(train_df) * args.val_ratio))) if len(train_df) > 1 else 0
        val_indices = train_df.sample(n=val_size, random_state=args.seed).index if val_size else []
        val_df = train_df.loc[val_indices].reset_index(drop=True) if val_size else train_df.head(0)
        train_df = train_df.drop(val_indices, errors="ignore").reset_index(drop=True)
        if args.max_val_samples and args.max_val_samples > 0:
            val_df = val_df.head(args.max_val_samples).reset_index(drop=True)

    _write_parquet_atomic(
        pd.DataFrame(_build_rows(train_df, "train", args.prompt_col, args.labels_col)),
        output_dir / "train.parquet",
    )
    _write_parquet_atomic(
        pd.DataFrame(_build_rows(val_df, "val", args.prompt_col, args.labels_col)),
        output_dir / "val.parquet",
    )

    print(f"Wrote {len(train_df)} train rows and {len(val_df)} val rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
