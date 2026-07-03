from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    instances: int = Field(gt=0)
    seed: int


class ExperimentScaleConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: Literal["pilot", "medium"]
    train: SplitConfig
    validation: SplitConfig
    test: SplitConfig
    batch_size: int = Field(gt=0)
    steps_per_epoch: int = Field(gt=0)
    epochs: int = Field(gt=0)


class RLHyperparameters(BaseModel):
    model_config = ConfigDict(frozen=True)

    learning_rate: float = Field(default=1.0e-4, gt=0)
    learning_rate_decay: float = Field(default=1.0, gt=0, le=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    optimizer: Literal["adam"] = "adam"
    baseline: Literal["rollout"] = "rollout"
    baseline_alpha: float = Field(default=0.05, ge=0, le=1)
    baseline_warmup_epochs: int = Field(default=1, ge=0)
    exp_baseline_beta: float = Field(default=0.8, ge=0, le=1)
    log_every: int = Field(default=25, gt=0)
    checkpoint_every: int = Field(default=5, gt=0)
    save_best: bool = True
    save_last: bool = True
    keep_last_k: int = Field(default=3, ge=0)
    progress_bar: bool = True


class SupervisedHyperparameters(BaseModel):
    model_config = ConfigDict(frozen=True)

    learning_rate: float = Field(default=1.0e-4, gt=0)
    learning_rate_decay: float = Field(default=1.0, gt=0, le=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    optimizer: Literal["adam"] = "adam"
    label_smoothing: float = Field(default=0.0, ge=0, lt=1)
    log_every: int = Field(default=25, gt=0)
    checkpoint_every: int = Field(default=5, gt=0)
    save_best: bool = True
    save_last: bool = True
    keep_last_k: int = Field(default=3, ge=0)
    progress_bar: bool = True


PILOT_SCALE = ExperimentScaleConfig(
    name="pilot",
    train=SplitConfig(instances=64_000, seed=1234),
    validation=SplitConfig(instances=10_000, seed=4321),
    test=SplitConfig(instances=10_000, seed=9999),
    batch_size=512,
    steps_per_epoch=125,
    epochs=100,
)

MEDIUM_SCALE = ExperimentScaleConfig(
    name="medium",
    train=SplitConfig(instances=256_000, seed=1234),
    validation=SplitConfig(instances=10_000, seed=4321),
    test=SplitConfig(instances=10_000, seed=9999),
    batch_size=512,
    steps_per_epoch=500,
    epochs=100,
)

RL_HYPERPARAMETERS = RLHyperparameters()
SUPERVISED_HYPERPARAMETERS = SupervisedHyperparameters()

SCALES: dict[str, ExperimentScaleConfig] = {
    PILOT_SCALE.name: PILOT_SCALE,
    MEDIUM_SCALE.name: MEDIUM_SCALE,
}
