import pandas as pd

from wgan_gp_augmentation.data.loading import load_rows, normalize_label


def test_normalize_known_xss_variants() -> None:
    assert normalize_label("  Web   Attack – XSS ") == "Web Attack - XSS"


def test_reservoir_cap_is_deterministic(tmp_path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"Feature": range(100), "Label": ["BENIGN"] * 100}).to_csv(path, index=False)
    first, _ = load_rows([path], "Label", ["BENIGN"], 17, 10, 42)
    second, _ = load_rows([path], "Label", ["BENIGN"], 29, 10, 42)
    assert sorted(first["Feature"]) == sorted(second["Feature"])
