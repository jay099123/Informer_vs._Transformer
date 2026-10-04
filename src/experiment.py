from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .data import PreparedPowerData, build_loaders
from .models import ModelConfig, build_model, count_trainable_parameters


@dataclass(frozen=True)
class ExperimentOptions:
    output_dir: Path
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    stride: int = 4
    seed: int = 42
    num_workers: int = 0
    alignment_input_length: int = 672
    maximum_train_samples: int | None = None
    maximum_validation_samples: int | None = None
    maximum_test_samples: int | None = None
    force_retrain: bool = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _decoder_input(context: torch.Tensor, prediction_length: int) -> torch.Tensor:
    zeros = torch.zeros(
        context.size(0), prediction_length, context.size(2), device=context.device
    )
    return torch.cat([context, zeros], dim=1)


def _move_batch(batch, device: torch.device):
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


def _forward(model, batch):
    encoder_values, encoder_marks, context, targets, decoder_marks = batch
    decoder_values = _decoder_input(context, targets.size(1))
    predictions = model(encoder_values, encoder_marks, decoder_values, decoder_marks)
    return predictions, targets


def _run_training_epoch(model, loader, optimizer, scaler, device: torch.device) -> float:
    model.train()
    total_squared_error = 0.0
    total_elements = 0
    for batch in tqdm(loader, leave=False):
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            predictions, targets = _forward(model, batch)
            loss = F.mse_loss(predictions, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_squared_error += F.mse_loss(
            predictions.detach(), targets, reduction="sum"
        ).item()
        total_elements += targets.numel()
    return total_squared_error / total_elements


@torch.inference_mode()
def _validation_loss(model, loader, device: torch.device) -> float:
    model.eval()
    total_squared_error = 0.0
    total_elements = 0
    for batch in tqdm(loader, leave=False):
        batch = _move_batch(batch, device)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            predictions, targets = _forward(model, batch)
        total_squared_error += F.mse_loss(predictions, targets, reduction="sum").item()
        total_elements += targets.numel()
    return total_squared_error / total_elements


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate_model(
    model,
    loader,
    prepared: PreparedPowerData,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()

    # One untimed warm-up removes one-off kernel initialization from the comparison.
    warmup_batch = _move_batch(next(iter(loader)), device)
    with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
        _forward(model, warmup_batch)
    _synchronize(device)

    absolute_error = 0.0
    squared_error = 0.0
    element_count = 0
    sample_count = 0
    inference_seconds = 0.0
    sample_prediction = None
    sample_target = None

    for batch in tqdm(loader, leave=False):
        batch = _move_batch(batch, device)
        _synchronize(device)
        started = time.perf_counter()
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            predictions, targets = _forward(model, batch)
        _synchronize(device)
        inference_seconds += time.perf_counter() - started

        predictions = predictions.float() * prepared.target_std + prepared.target_mean
        targets = targets.float() * prepared.target_std + prepared.target_mean
        errors = predictions - targets
        absolute_error += errors.abs().sum().item()
        squared_error += errors.square().sum().item()
        element_count += errors.numel()
        sample_count += errors.size(0)
        if sample_prediction is None:
            sample_prediction = predictions[0, :, 0].detach().cpu().numpy()
            sample_target = targets[0, :, 0].detach().cpu().numpy()

    mse = squared_error / element_count
    metrics = {
        "mae": absolute_error / element_count,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "inference_seconds": inference_seconds,
        "inference_ms_per_sample": 1000.0 * inference_seconds / sample_count,
        "test_samples": sample_count,
    }
    samples = {"prediction": sample_prediction, "target": sample_target}
    return metrics, samples


def _options_signature(options: ExperimentOptions) -> dict:
    signature = asdict(options)
    signature["output_dir"] = str(options.output_dir.resolve())
    signature.pop("force_retrain")
    return signature


def run_experiment(
    config: ModelConfig,
    prepared: PreparedPowerData,
    options: ExperimentOptions,
) -> tuple[dict, list[dict], dict[str, np.ndarray]]:
    seed_everything(options.seed)
    device = _device()
    checkpoint_dir = options.output_dir / "checkpoints"
    prediction_dir = options.output_dir / "predictions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{config.name}.pt"

    loaders = build_loaders(
        prepared,
        input_length=config.input_length,
        label_length=config.label_length,
        prediction_length=config.prediction_length,
        batch_size=options.batch_size,
        stride=options.stride,
        seed=options.seed,
        num_workers=options.num_workers,
        maximum_train_samples=options.maximum_train_samples,
        maximum_validation_samples=options.maximum_validation_samples,
        maximum_test_samples=options.maximum_test_samples,
        alignment_input_length=options.alignment_input_length,
    )
    train_loader, validation_loader, test_loader = loaders
    model = build_model(config).to(device)
    history = []
    total_training_seconds = 0.0
    peak_gpu_memory_mb = 0.0

    loaded = False
    if checkpoint_path.exists() and not options.force_retrain:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if (
            checkpoint.get("config") == asdict(config)
            and checkpoint.get("options") == _options_signature(options)
        ):
            model.load_state_dict(checkpoint["state_dict"])
            history = checkpoint["history"]
            total_training_seconds = checkpoint["total_training_seconds"]
            peak_gpu_memory_mb = checkpoint["peak_gpu_memory_mb"]
            loaded = True
            print(f"Loaded checkpoint: {checkpoint_path}")

    if not loaded:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=options.learning_rate,
            weight_decay=options.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for epoch in range(1, options.epochs + 1):
            started = time.perf_counter()
            train_mse_standardized = _run_training_epoch(
                model, train_loader, optimizer, scaler, device
            )
            validation_mse_standardized = _validation_loss(
                model, validation_loader, device
            )
            epoch_seconds = time.perf_counter() - started
            total_training_seconds += epoch_seconds
            record = {
                "epoch": epoch,
                "train_mse_standardized": train_mse_standardized,
                "validation_mse_standardized": validation_mse_standardized,
                "epoch_seconds": epoch_seconds,
            }
            history.append(record)
            print(
                f"{config.name:>34} | epoch {epoch:02d}/{options.epochs} | "
                f"train MSE={train_mse_standardized:.5f} | "
                f"val MSE={validation_mse_standardized:.5f} | {epoch_seconds:.1f}s"
            )

        if device.type == "cuda":
            peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        torch.save(
            {
                "config": asdict(config),
                "options": _options_signature(options),
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "history": history,
                "total_training_seconds": total_training_seconds,
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
            },
            checkpoint_path,
        )

    test_metrics, samples = evaluate_model(model, test_loader, prepared, device)
    np.savez_compressed(prediction_dir / f"{config.name}.npz", **samples)
    result = {
        "name": config.name,
        "model_type": config.model_type,
        "input_length": config.input_length,
        "prediction_length": config.prediction_length,
        "attention_type": config.attention_type,
        "distil": config.distil,
        "generative_decoder": config.generative_decoder,
        "parameters": count_trainable_parameters(model),
        "epochs": options.epochs,
        "average_epoch_seconds": total_training_seconds / options.epochs,
        "total_training_seconds": total_training_seconds,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "device": str(device),
        **test_metrics,
    }
    return result, history, samples

