from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExpConfig:
    name: str
    seed: int
    selected_labels: tuple[str, ...]
    minority_labels: tuple[str, ...]


@dataclass(frozen=True)
class DataConfig:
    label_column: str
    chunk_size: int
    max_rows_per_class: int | None
    test_fraction: float
    validation_fraction: float
    variance_threshold: float
    pca_variance: float


@dataclass(frozen=True)
class ClfConfig:
    hidden_layers: tuple[int, ...]
    batch_size: int
    learning_rate: float
    max_epochs: int
    patience: int
    alpha: float


@dataclass(frozen=True)
class WganConfig:
    hidden_layers: tuple[int, ...]
    batch_size: int
    latent_dim: int
    train_steps: int
    critic_steps: int
    learning_rate: float
    beta1: float
    beta2: float
    gradient_penalty: float
    target_train_count: int
    log_interval: int


@dataclass(frozen=True)
class RunConfig:
    experiment: ExpConfig
    data: DataConfig
    classifier: ClfConfig
    wgan: WganConfig


def load_config(path: str | Path) -> RunConfig:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    experiment = raw["experiment"]
    data = raw["data"]
    classifier = raw["classifier"]
    wgan = raw["wgan"]
    return RunConfig(
        experiment=ExpConfig(
            name=str(experiment["name"]),
            seed=int(experiment["seed"]),
            selected_labels=tuple(experiment["selected_labels"]),
            minority_labels=tuple(experiment["minority_labels"]),
        ),
        data=DataConfig(
            label_column=str(data["label_column"]),
            chunk_size=int(data["chunk_size"]),
            max_rows_per_class=(
                None if data.get("max_rows_per_class") is None else int(data["max_rows_per_class"])
            ),
            test_fraction=float(data["test_fraction"]),
            validation_fraction=float(data["validation_fraction"]),
            variance_threshold=float(data["variance_threshold"]),
            pca_variance=float(data["pca_variance"]),
        ),
        classifier=ClfConfig(
            hidden_layers=tuple(int(value) for value in classifier["hidden_layers"]),
            batch_size=int(classifier["batch_size"]),
            learning_rate=float(classifier["learning_rate"]),
            max_epochs=int(classifier["max_epochs"]),
            patience=int(classifier["patience"]),
            alpha=float(classifier["alpha"]),
        ),
        wgan=WganConfig(
            hidden_layers=tuple(int(value) for value in wgan["hidden_layers"]),
            batch_size=int(wgan["batch_size"]),
            latent_dim=int(wgan["latent_dim"]),
            train_steps=int(wgan["train_steps"]),
            critic_steps=int(wgan["critic_steps"]),
            learning_rate=float(wgan["learning_rate"]),
            beta1=float(wgan["beta1"]),
            beta2=float(wgan["beta2"]),
            gradient_penalty=float(wgan["gradient_penalty"]),
            target_train_count=int(wgan["target_train_count"]),
            log_interval=int(wgan["log_interval"]),
        ),
    )
