import numpy as np
import pandas as pd
from wgan_gp_augmentation.data.splitting import clean_rows, split_rows


def test_cleaning_removes_invalid_and_duplicate_rows() -> None:
    frame = pd.DataFrame({"a": [1.0, 1.0, np.inf, 2.0], "Label": ["A", "A", "A", "B"]})
    cleaned, report = clean_rows(frame, "Label")
    assert len(cleaned) == 2
    assert report["invalid_rows_removed"] == 1
    assert report["duplicate_rows_removed"] == 1


def test_conflicting_duplicate_features_are_all_removed() -> None:
    frame = pd.DataFrame({"a": [1.0, 1.0], "Label": ["A", "B"]})
    cleaned, report = clean_rows(frame, "Label")
    assert cleaned.empty
    assert report["conflicting_feature_vectors"] == 1
    assert report["conflicting_rows_removed"] == 2


def test_stratified_split_is_disjoint_and_complete() -> None:
    frame = pd.DataFrame({"row": range(60), "value": range(60), "Label": ["A"] * 30 + ["B"] * 30})
    splits = split_rows(frame, "Label", 0.15, 0.15, 42)
    identifiers = [set(part["row"]) for part in (splits.train, splits.validation, splits.test)]
    assert not (identifiers[0] & identifiers[1] or identifiers[0] & identifiers[2] or identifiers[1] & identifiers[2])
    assert set.union(*identifiers) == set(frame["row"])
    assert all(set(part["Label"]) == {"A", "B"} for part in (splits.train, splits.validation, splits.test))
