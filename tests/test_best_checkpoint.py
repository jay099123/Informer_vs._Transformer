"""Regression tests for checkpoint selection, reuse, and version invalidation."""

from dataclasses import asdict, replace
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import experiment
from src.models import ModelConfig


class SnapshotModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0]))
        self.register_buffer("running_value", torch.tensor([0.0]))


class BestCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = ModelConfig(
            name="checkpoint_test", model_type="informer", input_length=16,
            label_length=4, prediction_length=4, encoder_input_size=1,
            time_feature_size=2,
        )
        self.options = experiment.ExperimentOptions(
            output_dir=Path(self.directory.name), epochs=3,
        )
        self.prepared = SimpleNamespace()
        self.checkpoint_path = self.options.output_dir / "checkpoints" / "checkpoint_test.pt"

    def run_with_losses(self, losses, expected_weight, options=None):
        model = SnapshotModel()
        epoch = 0
        self.training_learning_rates = []

        def train(*args):
            nonlocal epoch
            epoch += 1
            self.training_learning_rates.append(args[2].param_groups[0]["lr"])
            with torch.no_grad():
                model.weight.fill_(epoch)
                model.running_value.fill_(epoch * 10)
            return float(epoch)

        def evaluate(evaluated_model, *args):
            self.assertEqual(evaluated_model.weight.item(), expected_weight)
            self.assertEqual(evaluated_model.running_value.item(), expected_weight * 10)
            return {"mse": 0.0}, {"prediction": np.zeros(4), "target": np.zeros(4)}

        with (
            patch.object(experiment, "_device", return_value=torch.device("cpu")),
            patch.object(experiment, "build_model", return_value=model),
            patch.object(experiment, "build_loaders", return_value=([], [], [])),
            patch.object(experiment, "_run_training_epoch", side_effect=train) as training,
            patch.object(experiment, "_validation_loss", side_effect=losses),
            patch.object(experiment, "evaluate_model", side_effect=evaluate),
        ):
            result, history, _ = experiment.run_experiment(
                self.config, self.prepared, options or self.options
            )
        checkpoint = torch.load(self.checkpoint_path, weights_only=False)
        return result, history, checkpoint, training.call_count

    def test_best_middle_epoch_is_saved_and_evaluated_then_reused(self):
        result, history, checkpoint, trained = self.run_with_losses([0.8, 0.3, 0.6], 2)
        self.assertEqual(trained, 3)
        self.assertEqual(len(history), 3)
        self.assertEqual(result["best_epoch"], 2)
        self.assertEqual(result["best_validation_mse_standardized"], 0.3)
        self.assertEqual(checkpoint["state_dict"]["weight"].item(), 2)
        self.assertEqual(checkpoint["state_dict"]["running_value"].item(), 20)
        self.assertEqual(checkpoint["checkpoint_version"], experiment.CHECKPOINT_VERSION)
        reused, reused_history, _, trained = self.run_with_losses([], 2)
        self.assertEqual(trained, 0)
        self.assertEqual(reused["best_epoch"], 2)
        self.assertEqual(reused_history, history)

    def test_first_last_and_tied_best_epochs(self):
        for losses, best in (([0.1, 0.2, 0.3], 1),
                             ([0.3, 0.2, 0.1], 3),
                             ([0.3, 0.1, 0.1], 2)):
            with self.subTest(losses=losses):
                options = replace(self.options, force_retrain=True)
                result, _, _, _ = self.run_with_losses(losses, best, options)
                self.assertEqual(result["best_epoch"], best)

    def test_legacy_checkpoint_is_not_reused(self):
        self.checkpoint_path.parent.mkdir(parents=True)
        # Even a matching config/options must not reuse old final-epoch weights.
        torch.save({"config": asdict(self.config),
                    "options": experiment._options_signature(self.options)}, self.checkpoint_path)
        result, _, _, trained = self.run_with_losses([0.8, 0.3, 0.6], 2)
        self.assertEqual(trained, 3)
        self.assertEqual(result["best_epoch"], 2)

    def test_nonfinite_validation_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "non-finite validation MSE at epoch 2"):
            self.run_with_losses([0.8, float("nan"), 0.3], 1)
        self.assertFalse(self.checkpoint_path.exists())

    def test_plateau_reduces_next_epoch_lr_and_respects_floor_then_reuses(self):
        options = replace(self.options, epochs=10, min_learning_rate=2.5e-5)
        result, history, checkpoint, trained = self.run_with_losses(
            [1.0] + [1.1] * 9, 1, options
        )
        used_rates = [1e-4] * 4 + [5e-5] * 3 + [2.5e-5] * 3
        next_rates = [1e-4] * 3 + [5e-5] * 3 + [2.5e-5] * 4
        self.assertEqual(trained, 10)
        self.assertEqual(self.training_learning_rates, used_rates)
        self.assertEqual([row["learning_rate"] for row in history], used_rates)
        self.assertEqual([row["next_learning_rate"] for row in history], next_rates)
        self.assertEqual(checkpoint["history"], history)
        self.assertEqual(result["initial_learning_rate"], 1e-4)
        self.assertEqual(result["final_learning_rate"], 2.5e-5)
        reused, reused_history, _, trained = self.run_with_losses([], 1, options)
        self.assertEqual(trained, 0)
        self.assertEqual(reused_history, history)
        self.assertEqual(reused["final_learning_rate"], 2.5e-5)

    def test_validation_improvement_resets_scheduler_patience(self):
        options = replace(self.options, epochs=7, lr_scheduler_patience=1)
        result, _, _, _ = self.run_with_losses(
            [1.0, 0.9, 0.91, 0.8, 0.81, 0.82, 0.7], 7, options
        )
        self.assertEqual(self.training_learning_rates, [1e-4] * 6 + [5e-5])
        self.assertEqual(result["best_epoch"], 7)

    def test_disabled_scheduler_keeps_learning_rate_constant(self):
        options = replace(self.options, epochs=6, use_lr_scheduler=False)
        with patch.object(torch.optim.lr_scheduler, "ReduceLROnPlateau") as scheduler:
            result, history, _, _ = self.run_with_losses([1.0] + [1.1] * 5, 1, options)
            scheduler.assert_not_called()
        self.assertEqual(self.training_learning_rates, [1e-4] * 6)
        self.assertTrue(all(row["next_learning_rate"] == 1e-4 for row in history))
        self.assertEqual(result["final_learning_rate"], 1e-4)

    def test_scheduler_changes_or_old_version_force_retraining(self):
        self.run_with_losses([0.8, 0.3, 0.6], 2)
        options = replace(self.options, lr_scheduler_factor=0.1)
        _, _, checkpoint, trained = self.run_with_losses([0.8, 0.3, 0.6], 2, options)
        self.assertEqual(trained, 3)
        checkpoint["checkpoint_version"] = 2
        torch.save(checkpoint, self.checkpoint_path)
        _, _, _, trained = self.run_with_losses([0.8, 0.3, 0.6], 2, options)
        self.assertEqual(trained, 3)

    def test_invalid_scheduler_settings_fail_before_building_loaders(self):
        for settings in (
            {"lr_scheduler_factor": 1.0},
            {"lr_scheduler_patience": -1},
            {"lr_scheduler_threshold": float("nan")},
            {"min_learning_rate": 2e-4},
        ):
            with self.subTest(settings=settings), patch.object(experiment, "build_loaders") as loaders:
                with self.assertRaises(ValueError):
                    experiment.run_experiment(
                        self.config, self.prepared, replace(self.options, **settings)
                    )
                loaders.assert_not_called()

    def test_real_training_and_checkpoint_reuse(self):
        rng = np.random.default_rng(42)
        prepared = SimpleNamespace(
            values=rng.standard_normal((96, 3)).astype(np.float32),
            time_marks=rng.standard_normal((96, 2)).astype(np.float32),
            target_index=0, train_end=48, validation_end=72,
            target_mean=1.0, target_std=1.2,
        )
        options = replace(self.options, epochs=2, batch_size=4, stride=4,
                          alignment_input_length=8)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for model_type, decoder in (("informer", True), ("informer", False),
                                        ("transformer", True)):
                with self.subTest(model_type=model_type, decoder=decoder):
                    config = replace(
                        self.config, name=f"smoke_{model_type}_{decoder}",
                        model_type=model_type, input_length=8,
                        encoder_input_size=3, d_model=8, n_heads=2, d_ff=16,
                        dropout=0.0, factor=1, generative_decoder=decoder,
                    )
                    with patch.object(experiment, "_device", return_value=torch.device("cpu")):
                        result, history, samples = experiment.run_experiment(config, prepared, options)
                        with patch.object(experiment, "_run_training_epoch") as training:
                            reused, reused_history, _ = experiment.run_experiment(config, prepared, options)
                            training.assert_not_called()
                    self.assertEqual(len(history), 2)
                    self.assertIn(result["best_epoch"], (1, 2))
                    self.assertEqual(result["best_epoch"], reused["best_epoch"])
                    self.assertEqual(history, reused_history)
                    self.assertTrue(np.isfinite(result["mse"]))
                    self.assertEqual(samples["prediction"].shape, (4,))
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
