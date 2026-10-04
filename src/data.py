from __future__ import annotations

import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset


KAGGLE_SLUG = "uciml/electric-power-consumption-data-set"
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
RAW_FILENAME = "household_power_consumption.txt"
NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
TARGET_COLUMN = "Global_active_power"


def download_dataset(data_dir: str | Path) -> Path:
    """Download the public Kaggle copy, with the UCI source as a fallback."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / RAW_FILENAME
    if target.exists():
        return target

    kaggle_error = None
    try:
        import kagglehub

        downloaded = Path(kagglehub.dataset_download(KAGGLE_SLUG))
        candidates = list(downloaded.rglob(RAW_FILENAME))
        if not candidates:
            raise FileNotFoundError(f"{RAW_FILENAME} was not found in {downloaded}")
        shutil.copy2(candidates[0], target)
        return target
    except Exception as error:  # Kaggle credentials/network may be unavailable.
        kaggle_error = error
        print(f"KaggleHub download unavailable ({error}); trying the UCI source.")

    archive = data_dir / "individual_household_power_consumption.zip"
    try:
        urllib.request.urlretrieve(UCI_ZIP_URL, archive)
        with zipfile.ZipFile(archive) as zip_file:
            member = next(
                name for name in zip_file.namelist() if name.endswith(RAW_FILENAME)
            )
            with zip_file.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    except Exception as uci_error:
        raise RuntimeError(
            "Automatic download failed. Download the Kaggle dataset manually and place "
            f"{RAW_FILENAME} in {data_dir.resolve()}. "
            f"Kaggle error: {kaggle_error}; UCI error: {uci_error}"
        ) from uci_error
    finally:
        if archive.exists():
            archive.unlink()
    return target


def _time_features(index: pd.DatetimeIndex) -> np.ndarray:
    minute_of_day = index.hour.to_numpy() * 60 + index.minute.to_numpy()
    day_of_week = index.dayofweek.to_numpy()
    day_of_year = index.dayofyear.to_numpy()
    return np.column_stack(
        [
            np.sin(2 * np.pi * minute_of_day / 1440.0),
            np.cos(2 * np.pi * minute_of_day / 1440.0),
            np.sin(2 * np.pi * day_of_week / 7.0),
            np.cos(2 * np.pi * day_of_week / 7.0),
            np.sin(2 * np.pi * day_of_year / 365.25),
            np.cos(2 * np.pi * day_of_year / 365.25),
        ]
    ).astype(np.float32)


def _load_and_resample(raw_path: Path, cache_path: Path, frequency: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    if cache_path.exists():
        cached = np.load(cache_path)
        values = cached["values"].astype(np.float32)
        timestamps = pd.to_datetime(cached["timestamps"])
        return values, pd.DatetimeIndex(timestamps)

    print("Reading the minute-level dataset. The first preprocessing run may take a few minutes.")
    raw = pd.read_csv(
        raw_path,
        sep=";",
        na_values=["?", ""],
        low_memory=False,
        usecols=["Date", "Time", *NUMERIC_COLUMNS],
    )
    timestamps = pd.to_datetime(
        raw.pop("Date").astype(str) + " " + raw.pop("Time").astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    raw.index = timestamps
    raw = raw.loc[~raw.index.isna(), NUMERIC_COLUMNS]
    raw = raw.apply(pd.to_numeric, errors="coerce").sort_index()

    # Mean aggregation preserves the physical units of power measurements.
    frame = raw.resample(frequency).mean()
    frame = frame.interpolate(method="time", limit_direction="both").ffill().bfill()
    if frame.isna().any().any():
        raise ValueError("Missing values remain after interpolation.")

    values = frame.to_numpy(dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        values=values,
        timestamps=frame.index.to_numpy(dtype="datetime64[ns]"),
    )
    return values, frame.index


@dataclass
class PreparedPowerData:
    values: np.ndarray
    time_marks: np.ndarray
    timestamps: pd.DatetimeIndex
    columns: tuple[str, ...]
    target_index: int
    target_mean: float
    target_std: float
    train_end: int
    validation_end: int
    frequency: str

    @property
    def input_size(self) -> int:
        return self.values.shape[1]

    @property
    def mark_size(self) -> int:
        return self.time_marks.shape[1]


def prepare_power_data(
    data_dir: str | Path,
    frequency: str = "15min",
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> PreparedPowerData:
    data_dir = Path(data_dir)
    raw_path = download_dataset(data_dir)
    cache_path = data_dir / f"processed_{frequency}.npz"
    values, timestamps = _load_and_resample(raw_path, cache_path, frequency)

    total = len(values)
    train_end = int(total * train_ratio)
    validation_end = int(total * (train_ratio + validation_ratio))
    train_values = values[:train_end]
    means = train_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    standard_deviations = train_values.std(axis=0, dtype=np.float64).astype(np.float32)
    standard_deviations = np.where(standard_deviations < 1e-6, 1.0, standard_deviations)
    standardized = ((values - means) / standard_deviations).astype(np.float32)

    target_index = NUMERIC_COLUMNS.index(TARGET_COLUMN)
    return PreparedPowerData(
        values=standardized,
        time_marks=_time_features(timestamps),
        timestamps=timestamps,
        columns=tuple(NUMERIC_COLUMNS),
        target_index=target_index,
        target_mean=float(means[target_index]),
        target_std=float(standard_deviations[target_index]),
        train_end=train_end,
        validation_end=validation_end,
        frequency=frequency,
    )


class PowerWindowDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        time_marks: np.ndarray,
        target_index: int,
        input_length: int,
        label_length: int,
        prediction_length: int,
        stride: int,
        start_offset: int = 0,
    ) -> None:
        if label_length > input_length:
            raise ValueError("label_length must not exceed input_length")
        self.values = values
        self.time_marks = time_marks
        self.target_index = target_index
        self.input_length = input_length
        self.label_length = label_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.start_offset = start_offset
        available = len(values) - start_offset - input_length - prediction_length + 1
        self.length = max(0, (available + stride - 1) // stride)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        start = self.start_offset + index * self.stride
        input_end = start + self.input_length
        prediction_end = input_end + self.prediction_length
        label_start = input_end - self.label_length

        encoder_values = self.values[start:input_end]
        encoder_marks = self.time_marks[start:input_end]
        decoder_context = self.values[
            label_start:input_end, self.target_index : self.target_index + 1
        ]
        targets = self.values[
            input_end:prediction_end, self.target_index : self.target_index + 1
        ]
        decoder_marks = self.time_marks[label_start:prediction_end]
        return tuple(
            torch.from_numpy(array)
            for array in (
                encoder_values,
                encoder_marks,
                decoder_context,
                targets,
                decoder_marks,
            )
        )


def _limit_dataset(dataset: Dataset, maximum: int | None) -> Dataset:
    if maximum is None or len(dataset) <= maximum:
        return dataset
    indices = np.linspace(0, len(dataset) - 1, maximum, dtype=np.int64).tolist()
    return Subset(dataset, indices)


def build_loaders(
    prepared: PreparedPowerData,
    input_length: int,
    label_length: int,
    prediction_length: int,
    batch_size: int,
    stride: int,
    seed: int,
    num_workers: int = 0,
    maximum_train_samples: int | None = None,
    maximum_validation_samples: int | None = None,
    maximum_test_samples: int | None = None,
    alignment_input_length: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    segments = [
        (0, prepared.train_end),
        (prepared.train_end, prepared.validation_end),
        (prepared.validation_end, len(prepared.values)),
    ]
    datasets = []
    for start, end in segments:
        datasets.append(
            PowerWindowDataset(
                prepared.values[start:end],
                prepared.time_marks[start:end],
                prepared.target_index,
                input_length,
                label_length,
                prediction_length,
                stride,
                start_offset=max(0, (alignment_input_length or input_length) - input_length),
            )
        )

    train_dataset = _limit_dataset(datasets[0], maximum_train_samples)
    validation_dataset = _limit_dataset(datasets[1], maximum_validation_samples)
    test_dataset = _limit_dataset(datasets[2], maximum_test_samples)
    if min(len(train_dataset), len(validation_dataset), len(test_dataset)) == 0:
        raise ValueError("A split is too short for the requested input/prediction lengths.")

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, validation_loader, test_loader
