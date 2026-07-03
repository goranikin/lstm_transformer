from pydantic import BaseModel, ConfigDict, Field


class PNModelConfig(BaseModel):
    """Pointer Network architecture for AM-paper benchmarks (Section 5).

    Uses a single-layer LSTM encoder with additive attention and one glimpse,
    matching the reference implementation in attention-learn-to-route.
    """

    model_config = ConfigDict(validate_assignment=True)

    hidden_size: int = Field(
        default=256,
        gt=0,
        description=(
            "256 gives a parameter count close to the default AM "
            "(d_h=128, n_layers=3)."
        ),
    )
    num_layers: int = Field(default=1, gt=0)
    dropout: float = Field(default=0.0, ge=0, le=1)
    tanh_clip: float = Field(default=10.0, ge=0)
    n_glimpses: int = Field(default=1, ge=0)
    mask_inner: bool = True
    mask_logits: bool = True
