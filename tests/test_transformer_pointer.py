import unittest
from typing import cast

import torch

from src.constants import ProblemName
from src.model import NCOModel
from src.models.decoders import TransformerPointerDecoder


class TransformerPointerDecoderTests(unittest.TestCase):
    def test_tsp_supervised_loss_backpropagates_through_transformer_cell(self) -> None:
        torch.manual_seed(7)
        model = _model(problem="tsp", input_dim=2)
        batch = {
            "loc": torch.rand(2, 5, 2),
            "target_actions": torch.tensor(
                [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=torch.long
            ),
        }

        loss = model.supervised_loss(batch)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsInstance(model.decoder, TransformerPointerDecoder)
        assert isinstance(model.decoder, TransformerPointerDecoder)
        attention_weight = dict(model.decoder.named_parameters())[
            "layers.0.self_attn.in_proj_weight"
        ]
        gradient = attention_weight.grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertIsNone(model.decoder._layer_histories)

    def test_knapsack_decode_supports_learned_stop_key(self) -> None:
        torch.manual_seed(11)
        model = _model(problem="knapsack", input_dim=2)
        batch = {
            "weights": torch.tensor([[2.0, 3.0, 4.0], [1.0, 2.0, 5.0]]),
            "values": torch.tensor([[3.0, 4.0, 5.0], [2.0, 4.0, 7.0]]),
            "capacity": torch.tensor([5.0, 4.0]),
        }

        output = model(batch)

        self.assertEqual(tuple(output.actions.shape[:1]), (2,))
        self.assertTrue(bool(output.feasible.all()))
        self.assertIsInstance(model.decoder, TransformerPointerDecoder)
        assert isinstance(model.decoder, TransformerPointerDecoder)
        self.assertIsNone(model.decoder._layer_histories)


def _model(*, problem: str, input_dim: int) -> NCOModel:
    return NCOModel(
        problem=cast(ProblemName, problem),
        encoder_kind="attention",
        decoder_kind="transformer_pointer",
        input_dim=input_dim,
        d_model=16,
        num_layers=1,
        num_heads=4,
        d_ff=32,
        transformer_decoder_layers=1,
        dropout=0.0,
    )


if __name__ == "__main__":
    unittest.main()
