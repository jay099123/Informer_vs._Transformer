"""Run with: python -m unittest discover -s tests -v (from Informer)."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch


# Load the model module without importing dataset/download dependencies from src.
spec = importlib.util.spec_from_file_location(
    "informer_models", Path(__file__).resolve().parents[1] / "src" / "models.py"
)
models = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = models
spec.loader.exec_module(models)


class AttentionDtypeTests(unittest.TestCase):
    def test_float32_matches_full_attention_when_all_queries_are_selected(self):
        for causal in (False, True):
            with self.subTest(causal=causal):
                torch.manual_seed(42)
                inputs = [torch.randn(2, 12, 2, 4) for _ in range(3)]
                sparse_inputs = [x.clone().requires_grad_() for x in inputs]
                full_inputs = [x.clone().requires_grad_() for x in inputs]
                actual = models.ProbSparseAttention(factor=100, dropout=0.0)(
                    *sparse_inputs, causal=causal
                )
                expected = models.FullAttention(dropout=0.0)(*full_inputs, causal=causal)
                torch.testing.assert_close(actual, expected)
                actual.square().sum().backward()
                expected.square().sum().backward()
                for actual_input, expected_input in zip(sparse_inputs, full_inputs):
                    torch.testing.assert_close(actual_input.grad, expected_input.grad)

    def test_cpu_autocast_with_float32_context(self):
        # Autocast matmul returns bfloat16, while the mean/cumsum context
        # remains float32. This reproduces the scatter dtype mismatch on CPU.
        for causal in (False, True):
            with self.subTest(causal=causal):
                torch.manual_seed(42)
                inputs = [torch.randn(2, 12, 2, 4, requires_grad=True) for _ in range(3)]
                with torch.amp.autocast("cpu", dtype=torch.bfloat16):
                    output = models.ProbSparseAttention(factor=1, dropout=0.0)(
                        *inputs, causal=causal
                    )
                self.assertEqual(output.dtype, torch.float32)
                self.assertEqual(output.shape, inputs[0].shape)
                self.assertTrue(torch.isfinite(output).all().item())
                output.square().mean().backward()
                for tensor in inputs:
                    self.assertIsNotNone(tensor.grad)
                    self.assertTrue(torch.isfinite(tensor.grad).all().item())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_autocast_informer_forward_and_backward(self):
        # Exercise both encoder mean context and decoder float32 cumsum
        # using half-precision projections, as in the notebook training loop.
        torch.manual_seed(42)
        config = models.ModelConfig(
            name="amp_regression", model_type="informer", input_length=16,
            label_length=4, prediction_length=4, encoder_input_size=3,
            time_feature_size=2, d_model=16, n_heads=2, encoder_layers=2,
            decoder_layers=1, d_ff=32, dropout=0.0, factor=1,
        )
        model = models.build_model(config).cuda()
        inputs = [torch.randn(*shape, device="cuda") for shape in (
            (2, 16, 3), (2, 16, 2), (2, 8, 1), (2, 8, 2)
        )]
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(*inputs)
            loss = torch.nn.functional.mse_loss(output, torch.zeros_like(output))
        self.assertEqual(output.shape, (2, 4, 1))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())


class OfficialAlignmentTests(unittest.TestCase):
    def test_probsparse_selects_queries_using_full_key_length(self):
        # With two sampled keys, the official scores select queries 0 and 2.
        # Dividing by the sample count instead selects query 1.
        queries = torch.zeros(1, 7, 1, 2)
        queries[0, :3, 0] = torch.tensor([[10.0, 9.0], [5.0, 0.0], [9.0, 8.0]])
        keys = torch.zeros_like(queries)
        keys[0, :2, 0] = torch.eye(2)
        values = torch.arange(14.0).reshape(1, 7, 1, 2)
        sampled_indices = torch.tensor([[0, 1]]).expand(7, 2)
        with patch.object(torch, "randint", return_value=sampled_indices) as sample:
            actual = models.ProbSparseAttention(factor=1, dropout=0.0)(
                queries, keys, values
            )
        sample.assert_called_once_with(7, (7, 2), device=queries.device)
        full = models.FullAttention(dropout=0.0)(queries, keys, values)
        expected = values.mean(dim=1, keepdim=True).expand_as(values).clone()
        expected[:, [0, 2]] = full[:, [0, 2]]
        torch.testing.assert_close(actual, expected)

    def test_mix_preserves_official_transpose_before_reshape(self):
        attention_output = torch.arange(12.0).reshape(1, 3, 2, 2)

        class FixedAttention(torch.nn.Module):
            def forward(self, queries, keys, values, causal=False):
                return attention_output

        for mix in (False, True):
            with self.subTest(mix=mix):
                layer = models.AttentionLayer(FixedAttention(), 4, 2, mix=mix)
                with torch.no_grad():
                    layer.output_projection.weight.copy_(torch.eye(4))
                    layer.output_projection.bias.zero_()
                inputs = torch.zeros(1, 3, 4)
                actual = layer(inputs, inputs, inputs)
                if mix:
                    expected = torch.tensor([[[0., 1., 4., 5.],
                                              [8., 9., 2., 3.],
                                              [6., 7., 10., 11.]]])
                else:
                    expected = torch.arange(12.0).reshape(1, 3, 4)
                torch.testing.assert_close(actual, expected)

    def test_mix_is_only_enabled_for_decoder_self_attention(self):
        for mix in (False, True):
            with self.subTest(mix=mix):
                config = models.ModelConfig(
                    name="mix_wiring", model_type="informer", input_length=16,
                    label_length=4, prediction_length=4, encoder_input_size=3,
                    time_feature_size=2, d_model=16, n_heads=2, mix=mix,
                )
                model = models.Informer(config)
                for layer in model.encoder.layers:
                    self.assertFalse(layer.self_attention.mix)
                for layer in model.decoder_layers:
                    self.assertEqual(layer.self_attention.mix, mix)
                    self.assertFalse(layer.cross_attention.mix)


if __name__ == "__main__":
    unittest.main()
