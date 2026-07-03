from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AMModelConfig(BaseModel):
    """Hyperparameters from Section 5 of the paper."""

    model_config = ConfigDict(validate_assignment=True)

    d_h: int = Field(default=128, gt=0)
    n_layers: int = Field(default=3, gt=0)
    n_heads: int = Field(default=8, gt=0)
    d_ff: int = Field(default=512, gt=0)
    tanh_clip: float = Field(default=10.0, ge=0)
    normalization: Literal["batch", "instance"] = "batch"

    @model_validator(mode="after")
    def validate_attention_dimensions(self) -> Self:
        if self.d_h % self.n_heads != 0:
            raise ValueError("d_h must be divisible by n_heads")
        return self
