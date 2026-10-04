from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src import ExperimentOptions, ModelConfig, prepare_power_data, run_experiment


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_LENGTHS = (96, 336, 672)
LABEL_LENGTH = 48
PREDICTION_LENGTH = 96


def build_experiment_plan(
    encoder_input_size: int,
    time_feature_size: int,
    input_lengths=DEFAULT_INPUT_LENGTHS,
) -> list[ModelConfig]:
    common = dict(
        label_length=LABEL_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        encoder_input_size=encoder_input_size,
        time_feature_size=time_feature_size,
        d_model=64,
        n_heads=4,
        encoder_layers=2,
        decoder_layers=1,
        d_ff=128,
        dropout=0.1,
        factor=5,
    )
    plan = []
    for input_length in input_lengths:
        plan.append(
            ModelConfig(
                name=f"transformer_L{input_length}",
                model_type="transformer",
                input_length=input_length,
                attention_type="full",
                distil=False,
                generative_decoder=True,
                **common,
            )
        )
        plan.append(
            ModelConfig(
                name=f"informer_L{input_length}",
                model_type="informer",
                input_length=input_length,
                attention_type="prob",
                distil=True,
                generative_decoder=True,
                **common,
            )
        )

    longest = max(input_lengths)
    plan.extend(
        [
            ModelConfig(
                name=f"ablate_probsparse_L{longest}",
                model_type="informer",
                input_length=longest,
                attention_type="full",
                distil=True,
                generative_decoder=True,
                **common,
            ),
            ModelConfig(
                name=f"ablate_distilling_L{longest}",
                model_type="informer",
                input_length=longest,
                attention_type="prob",
                distil=False,
                generative_decoder=True,
                **common,
            ),
            ModelConfig(
                name=f"ablate_generative_decoder_L{longest}",
                model_type="informer",
                input_length=longest,
                attention_type="prob",
                distil=True,
                generative_decoder=False,
                **common,
            ),
        ]
    )
    return plan


def _save_loss_curves(histories: dict[str, list[dict]], figure_dir: Path) -> None:
    columns = 3
    rows = int(np.ceil(len(histories) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows), squeeze=False)
    for axis, (name, history) in zip(axes.flat, histories.items()):
        frame = pd.DataFrame(history)
        axis.plot(frame["epoch"], frame["train_mse_standardized"], label="training")
        axis.plot(
            frame["epoch"], frame["validation_mse_standardized"], label="validation"
        )
        axis.set_title(name)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Standardized MSE")
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in axes.flat[len(histories) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(figure_dir / "all_loss_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_prediction_curves(samples: dict[str, dict[str, np.ndarray]], figure_dir: Path) -> None:
    columns = 3
    rows = int(np.ceil(len(samples) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows), squeeze=False)
    for axis, (name, sample) in zip(axes.flat, samples.items()):
        axis.plot(sample["target"], label="actual", linewidth=2)
        axis.plot(sample["prediction"], label="prediction", linewidth=1.6)
        axis.set_title(name)
        axis.set_xlabel("15-minute forecast step")
        axis.set_ylabel("Global active power (kW)")
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in axes.flat[len(samples) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(figure_dir / "all_predictions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_long_sequence_comparison(results: pd.DataFrame, figure_dir: Path) -> None:
    main = results[results["name"].str.match(r"^(transformer|informer)_L")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for axis, metric, title in zip(
        axes,
        ("mae", "mse", "rmse"),
        ("MAE ↓", "MSE ↓", "RMSE ↓"),
    ):
        for model_type, group in main.groupby("model_type"):
            group = group.sort_values("input_length")
            axis.plot(group["input_length"], group[metric], marker="o", label=model_type)
        axis.set_title(title)
        axis.set_xlabel("Input Length")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "long_sequence_accuracy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    efficiency_metrics = [
        ("average_epoch_seconds", "Seconds per epoch ↓"),
        ("peak_gpu_memory_mb", "Peak GPU memory (MB) ↓"),
        ("inference_ms_per_sample", "Inference ms/sample ↓"),
        ("parameters", "Trainable parameters ↓"),
    ]
    for axis, (metric, title) in zip(axes.flat, efficiency_metrics):
        for model_type, group in main.groupby("model_type"):
            group = group.sort_values("input_length")
            axis.plot(group["input_length"], group[metric], marker="o", label=model_type)
        axis.set_title(title)
        axis.set_xlabel("Input Length")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "long_sequence_efficiency.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_ablation_comparison(results: pd.DataFrame, figure_dir: Path) -> None:
    longest = int(results["input_length"].max())
    names = [
        f"informer_L{longest}",
        f"ablate_probsparse_L{longest}",
        f"ablate_distilling_L{longest}",
        f"ablate_generative_decoder_L{longest}",
    ]
    ablation = results.set_index("name").loc[names]
    labels = ["Full Informer", "No ProbSparse", "No Distilling", "No Generative Decoder"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, metric, title in zip(
        axes,
        ("mae", "rmse", "inference_ms_per_sample"),
        ("MAE ↓", "RMSE ↓", "Inference ms/sample ↓"),
    ):
        axis.bar(labels, ablation[metric], color=["#4C78A8", "#F58518", "#E45756", "#72B7B2"])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "informer_ablation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_summary_figures(
    results: pd.DataFrame,
    histories: dict[str, list[dict]],
    samples: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _save_loss_curves(histories, figure_dir)
    _save_prediction_curves(samples, figure_dir)
    _save_long_sequence_comparison(results, figure_dir)
    _save_ablation_comparison(results, figure_dir)


def run_all(quick_mode: bool = False, force_retrain: bool = False) -> pd.DataFrame:
    data_dir = PROJECT_DIR / "data"
    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_power_data(data_dir)
    plan = build_experiment_plan(prepared.input_size, prepared.mark_size)

    options = ExperimentOptions(
        output_dir=output_dir,
        epochs=1 if quick_mode else 10,
        batch_size=8 if quick_mode else 32,
        learning_rate=1e-4,
        weight_decay=1e-5,
        stride=4,
        seed=42,
        num_workers=0,
        alignment_input_length=max(DEFAULT_INPUT_LENGTHS),
        maximum_train_samples=256 if quick_mode else None,
        maximum_validation_samples=64 if quick_mode else None,
        maximum_test_samples=64 if quick_mode else None,
        force_retrain=force_retrain,
    )
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
    print("Quick mode:", quick_mode)
    print("Experiments:", len(plan))

    results = []
    histories = {}
    samples = {}
    for position, config in enumerate(plan, start=1):
        print(f"\n[{position}/{len(plan)}] {config.name}")
        result, history, sample = run_experiment(config, prepared, options)
        results.append(result)
        histories[config.name] = history
        samples[config.name] = sample

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(output_dir / "results.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "histories.json").open("w", encoding="utf-8") as file:
        json.dump(histories, file, ensure_ascii=False, indent=2)
    manifest = {
        "dataset": "Household Electric Power Consumption",
        "target": "Global_active_power",
        "frequency": prepared.frequency,
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "standardization_fit": "training only",
        "input_lengths": list(DEFAULT_INPUT_LENGTHS),
        "label_length": LABEL_LENGTH,
        "prediction_length": PREDICTION_LENGTH,
        "quick_mode": quick_mode,
        "options": {**asdict(options), "output_dir": str(output_dir.resolve())},
        "models": [asdict(config) for config in plan],
    }
    with (output_dir / "experiment_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, default=str)
    save_summary_figures(results_frame, histories, samples, output_dir)
    return results_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a small smoke experiment")
    parser.add_argument("--force", action="store_true", help="Ignore compatible checkpoints")
    arguments = parser.parse_args()
    results = run_all(quick_mode=arguments.quick, force_retrain=arguments.force)
    print("\nResults saved to:", (PROJECT_DIR / "outputs").resolve())
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

