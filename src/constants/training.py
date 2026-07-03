from typing import Literal, TypeAlias

ModeName: TypeAlias = Literal["supervised", "rl"]

MatrixName: TypeAlias = Literal[
    "supervised-smoke",
    "supervised",
    "rl",
    "module-test",
]

MATRIX_NAMES: tuple[MatrixName, ...] = (
    "supervised-smoke",
    "supervised",
    "rl",
    "module-test",
)

OptimizerName: TypeAlias = Literal["adam", "sgd"]

BaselineName: TypeAlias = Literal["rollout", "exponential"]
