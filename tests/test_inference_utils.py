from inference.generate_sana import safe_stem, stable_seed


def test_stable_seed_is_deterministic():
    first = stable_seed(42, "row-1", "Small left pleural effusion.")
    second = stable_seed(42, "row-1", "Small left pleural effusion.")
    assert first == second


def test_stable_seed_changes_with_row():
    assert stable_seed(42, "row-1", "prompt") != stable_seed(42, "row-2", "prompt")


def test_safe_stem_removes_path_separators():
    assert safe_stem("study/series/image", "fallback") == "study_series_image"
