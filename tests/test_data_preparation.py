from llm_sana.data.prepare_llavarad_prompt_parquet import _build_rows, _parse_labels

import pandas as pd


def test_parse_label_literal():
    assert _parse_labels("{'Edema': 1, 'No Finding': 0}") == {"Edema": 1, "No Finding": 0}


def test_build_rows_keeps_training_ground_truth():
    frame = pd.DataFrame(
        [
            {
                "id": "example",
                "annotated_prompt": "Mild pulmonary edema.",
                "chexpert_labels": "{'Edema': 1}",
            }
        ]
    )
    rows = _build_rows(frame, "train", "annotated_prompt", "chexpert_labels")
    assert rows[0]["reward_model"]["ground_truth"]["original_prompt"] == "Mild pulmonary edema."
    assert rows[0]["reward_model"]["ground_truth"]["chexpert_labels"] == {"Edema": 1}
