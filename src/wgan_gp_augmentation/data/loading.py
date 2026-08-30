from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LABEL_ALIASES = {
    "Web Attack – XSS": "Web Attack - XSS",
    "Web Attack � XSS": "Web Attack - XSS",
    "Web Attack – Brute Force": "Web Attack - Brute Force",
    "Web Attack � Brute Force": "Web Attack - Brute Force",
    "Web Attack – Sql Injection": "Web Attack - SQL Injection",
    "Web Attack � Sql Injection": "Web Attack - SQL Injection",
}


@dataclass(frozen=True)
class DataReport:
    files: tuple[str, ...]
    file_sha256: dict[str, str]
    rows_scanned: int
    rows_selected_before_cap: dict[str, int]
    rows_retained: dict[str, int]


def normalize_label(value: object) -> str:
    label = " ".join(str(value).strip().split())
    return LABEL_ALIASES.get(label, label)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _reservoir_update(
    reservoir: pd.DataFrame | None,
    incoming: pd.DataFrame,
    seen_before: int,
    capacity: int | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if capacity is None:
        return incoming.copy() if reservoir is None else pd.concat([reservoir, incoming], ignore_index=True)
    current = [] if reservoir is None else reservoir.to_dict("records")
    for offset, record in enumerate(incoming.to_dict("records"), start=1):
        seen = seen_before + offset
        if len(current) < capacity:
            current.append(record)
        else:
            position = int(rng.integers(0, seen))
            if position < capacity:
                current[position] = record
    return pd.DataFrame.from_records(current, columns=incoming.columns)


def load_rows(
    paths: Iterable[str | Path],
    label_column: str,
    selected_labels: Iterable[str],
    chunk_size: int,
    max_rows_per_class: int | None,
    seed: int,
) -> tuple[pd.DataFrame, DataReport]:
    csv_paths = tuple(Path(path).resolve() for path in paths)
    if not csv_paths:
        raise ValueError("At least one CSV path is required")
    wanted = set(selected_labels)
    reservoirs: dict[str, pd.DataFrame | None] = {label: None for label in wanted}
    counts = {label: 0 for label in wanted}
    rngs = {
        label: np.random.default_rng(seed + index)
        for index, label in enumerate(sorted(wanted))
    }
    scanned = 0
    expected_columns: list[str] | None = None
    for path in csv_paths:
        for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
            chunk.columns = [str(column).strip() for column in chunk.columns]
            if label_column not in chunk:
                raise ValueError(f"Missing label column {label_column!r} in {path}")
            if expected_columns is None:
                expected_columns = list(chunk.columns)
            elif list(chunk.columns) != expected_columns:
                raise ValueError(f"CSV schema differs in {path}")
            scanned += len(chunk)
            chunk[label_column] = chunk[label_column].map(normalize_label)
            chunk = chunk[chunk[label_column].isin(wanted)]
            for label, group in chunk.groupby(label_column, sort=False):
                before = counts[label]
                reservoirs[label] = _reservoir_update(
                    reservoirs[label], group, before, max_rows_per_class, rngs[label]
                )
                counts[label] += len(group)
    missing = sorted(label for label, count in counts.items() if count == 0)
    if missing:
        raise ValueError(f"Selected labels absent from input: {missing}")
    frame = pd.concat([reservoirs[label] for label in sorted(wanted)], ignore_index=True)
    report = DataReport(
        files=tuple(path.name for path in csv_paths),
        file_sha256={path.name: sha256_file(path) for path in csv_paths},
        rows_scanned=scanned,
        rows_selected_before_cap=dict(sorted(counts.items())),
        rows_retained={
            label: int((frame[label_column] == label).sum()) for label in sorted(wanted)
        },
    )
    return frame, report
