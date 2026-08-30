import numpy as np
import pandas as pd

from wgan_gp_augmentation.data.preprocessing import Preprocessor, encode_splits
from wgan_gp_augmentation.data.splitting import Splits


def test_preprocessor_fits_only_training_rows() -> None:
    train = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 3.0, 2.0, 4.0], "constant": [7] * 4, "Label": ["A", "A", "B", "B"]})
    validation = pd.DataFrame({"a": [100.0, 101.0], "b": [50.0, 51.0], "constant": [7, 7], "Label": ["A", "B"]})
    test = validation.copy()
    splits = Splits(train, validation, test)
    processor = Preprocessor(1e-12, 0.95).fit(train, "Label")
    arrays = encode_splits(splits, processor, "Label")
    assert processor.fit_row_count_ == len(train)
    assert processor.scaler.mean_[0] == np.mean(train["a"])
    assert processor.variance_filter.get_support().tolist() == [True, True, False]
    assert arrays["x_test"].shape[0] == 2
