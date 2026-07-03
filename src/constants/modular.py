from dataclasses import dataclass
from typing import Literal, TypeAlias

from src.constants.architecture import DecoderKind, EncoderKind
from src.constants.problems import ProblemName

ModularModelName: TypeAlias = Literal[
    "modular_pn",
    "modular_am",
    "am_lstm_pointer",
    "am_gru_pointer",
    "am_lstm_subset",
    "am_sigmoid_subset",
]

POINTER_DECODER_KINDS: frozenset[DecoderKind] = frozenset(
    {
        "attention_pointer",
        "lstm_pointer",
        "gru_pointer",
    }
)
SUBSET_DECODER_KINDS: frozenset[DecoderKind] = frozenset(
    {
        "attention_subset",
        "lstm_subset",
        "gru_subset",
        "sigmoid_subset",
    }
)


@dataclass(frozen=True)
class ModularArchitecture:
    name: ModularModelName
    encoder: EncoderKind
    decoder: DecoderKind
    description: str


MODULAR_ARCHITECTURES: tuple[ModularArchitecture, ...] = (
    ModularArchitecture(
        name="modular_pn",
        encoder="lstm",
        decoder="lstm_pointer",
        description="Pointer Network baseline (LSTM encoder + LSTM pointer decoder)",
    ),
    ModularArchitecture(
        name="modular_am",
        encoder="attention",
        decoder="attention_pointer",
        description="Attention Model baseline (graph attention + attention pointer)",
    ),
    ModularArchitecture(
        name="am_lstm_pointer",
        encoder="attention",
        decoder="lstm_pointer",
        description="Attention encoder with recurrent LSTM pointer decoder",
    ),
    ModularArchitecture(
        name="am_gru_pointer",
        encoder="attention",
        decoder="gru_pointer",
        description="Attention encoder with recurrent GRU pointer decoder",
    ),
    ModularArchitecture(
        name="am_lstm_subset",
        encoder="attention",
        decoder="lstm_subset",
        description="Attention encoder with autoregressive LSTM subset decoder",
    ),
    ModularArchitecture(
        name="am_sigmoid_subset",
        encoder="attention",
        decoder="sigmoid_subset",
        description="Attention encoder with parallel sigmoid subset head",
    ),
)

MODULAR_ARCHITECTURE_BY_NAME: dict[ModularModelName, ModularArchitecture] = {
    architecture.name: architecture for architecture in MODULAR_ARCHITECTURES
}

DEFAULT_MATRIX_MODELS: tuple[ModularModelName, ...] = tuple(
    architecture.name for architecture in MODULAR_ARCHITECTURES
)

MODULAR_MODEL_NAMES = frozenset(DEFAULT_MATRIX_MODELS)

# Canonical smoke problem for each architecture in the module-test matrix.
MODULE_TEST_PROBLEM: dict[ModularModelName, ProblemName] = {
    "modular_pn": "tsp",
    "modular_am": "tsp",
    "am_lstm_pointer": "cvrp",
    "am_gru_pointer": "orienteering",
    "am_lstm_subset": "mis",
    "am_sigmoid_subset": "knapsack",
}


def is_modular_model(model_name: str) -> bool:
    return model_name in MODULAR_ARCHITECTURE_BY_NAME


def modular_architecture(model_name: ModularModelName) -> ModularArchitecture:
    return MODULAR_ARCHITECTURE_BY_NAME[model_name]


def modular_encoder_decoder(
    model_name: ModularModelName,
) -> tuple[EncoderKind, DecoderKind]:
    architecture = modular_architecture(model_name)
    return architecture.encoder, architecture.decoder
