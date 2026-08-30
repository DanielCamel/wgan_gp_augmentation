import numpy as np

from wgan_gp_augmentation.augmentation import add_synthetic


def test_augmentation_changes_only_supplied_training_arrays() -> None:
    x_train = np.zeros((3, 2), dtype=np.float32)
    y_train = np.zeros(3, dtype=np.int64)
    x_validation = np.ones((2, 2), dtype=np.float32)
    x_test = np.full((2, 2), 2.0, dtype=np.float32)
    augmented_x, augmented_y = add_synthetic(
        x_train, y_train, np.ones((2, 2), dtype=np.float32), np.ones(2, dtype=np.int64)
    )
    assert augmented_x.shape == (5, 2)
    assert augmented_y.tolist() == [0, 0, 0, 1, 1]
    assert np.array_equal(x_validation, np.ones((2, 2)))
    assert np.array_equal(x_test, np.full((2, 2), 2.0))
