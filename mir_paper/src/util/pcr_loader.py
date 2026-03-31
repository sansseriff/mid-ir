from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
import csv
import pickle
import re

import numpy as np

from util.load_Pc230518 import clean_data, load_pcr_file


THRESHOLD_COLUMN_RE = re.compile(
    r"^(Counts|DCounts|ClicksOn|ClicksOff)_TL(?P<index>\d+)\((?P<value>[^)]+)\)$"
)


def _coerce_metadata_value(value: str) -> Any:
    value = value.strip()
    if not value:
        return value
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _as_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class ThresholdOption:
    index: int
    value: float

    @property
    def label(self) -> str:
        return f"TL{self.index} ({self.value:.3f})"


@dataclass
class PcrCurve:
    bias_current: np.ndarray
    counts: np.ndarray
    dark_counts: np.ndarray | None = None
    counts_error: np.ndarray | None = None
    dark_counts_error: np.ndarray | None = None
    threshold: ThresholdOption | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: str = ""
    label: str | None = None
    scale_factor: float = 1.0

    @property
    def bias_uA(self) -> np.ndarray:
        return self.bias_current * 1e6

    @property
    def normalized_counts(self) -> np.ndarray:
        peak = np.nanmax(self.counts)
        if not np.isfinite(peak) or peak == 0:
            return self.counts.copy()
        return self.counts / peak


@dataclass
class ThresholdSweepDataset:
    path: Path
    bias_current: np.ndarray
    columns: Dict[str, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_csv(cls, path: str | Path) -> "ThresholdSweepDataset":
        csv_path = _as_path(path)
        lines = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        metadata: Dict[str, Any] = {}
        header_index = 0

        if lines and lines[0].strip() == "# metadata,value":
            for idx, line in enumerate(lines[1:], start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("Bias_Current"):
                    header_index = idx
                    break
                key, value = next(csv.reader([line]))
                metadata[key] = _coerce_metadata_value(value)
        else:
            for idx, line in enumerate(lines):
                if line.strip().startswith("Bias_Current"):
                    header_index = idx
                    break

        rows = list(csv.reader(lines[header_index:]))
        if not rows:
            raise ValueError(f"No CSV rows found in {csv_path}")
        header = rows[0]
        numeric_rows = [
            row for row in rows[1:] if row and any(item.strip() for item in row)
        ]
        matrix = np.asarray(
            [[float(item) for item in row] for row in numeric_rows], dtype=float
        )
        columns = {name: matrix[:, idx] for idx, name in enumerate(header)}
        return cls(
            path=csv_path,
            bias_current=columns["Bias_Current"],
            columns=columns,
            metadata=metadata,
        )

    def threshold_options(self) -> list[ThresholdOption]:
        options: dict[int, ThresholdOption] = {}
        for column_name in self.columns:
            match = THRESHOLD_COLUMN_RE.match(column_name)
            if match is None:
                continue
            index = int(match.group("index"))
            value = float(match.group("value"))
            options[index] = ThresholdOption(index=index, value=value)
        return [options[idx] for idx in sorted(options)]

    def threshold_map(self) -> dict[int, float]:
        return {option.index: option.value for option in self.threshold_options()}

    def _resolve_threshold(self, selector: int | float | str | None) -> ThresholdOption:
        options = self.threshold_options()
        if not options:
            raise ValueError(f"No threshold columns found in {self.path}")
        if selector is None or selector == "auto":
            best = max(
                options,
                key=lambda option: np.nanmax(
                    self.columns[f"Counts_TL{option.index}({option.value:.3f})"]
                ),
            )
            return best
        if isinstance(selector, str):
            stripped = selector.strip()
            if stripped.upper().startswith("TL"):
                selector = int(stripped[2:])
            else:
                selector = float(stripped)
        if isinstance(selector, int):
            for option in options:
                if option.index == selector:
                    return option
            raise KeyError(f"Threshold index TL{selector} not found in {self.path}")
        selector_value = float(selector)
        return min(options, key=lambda option: abs(option.value - selector_value))

    def get_column(
        self, prefix: str, selector: int | float | str | None = None
    ) -> np.ndarray:
        option = self._resolve_threshold(selector)
        key = f"{prefix}_TL{option.index}({option.value:.3f})"
        if key not in self.columns:
            raise KeyError(f"Column {key} not found in {self.path}")
        return self.columns[key]

    def get_curve(
        self,
        selector: int | float | str | None = None,
        *,
        label: str | None = None,
    ) -> PcrCurve:
        option = self._resolve_threshold(selector)
        counts = self.get_column("Counts", option.index)
        dark_counts = None
        dark_key = f"DCounts_TL{option.index}({option.value:.3f})"
        if dark_key in self.columns:
            dark_counts = self.columns[dark_key]
        return PcrCurve(
            bias_current=self.bias_current,
            counts=counts,
            dark_counts=dark_counts,
            threshold=option,
            metadata=dict(self.metadata),
            source_path=str(self.path),
            label=label,
        )


@dataclass
class LegacyPcrDataset:
    path: Path
    columns: Dict[str, Any]
    cleaned: Dict[str, Any]
    integrations_per_setting: int

    @classmethod
    def from_text(
        cls,
        path: str | Path,
        *,
        integrations_per_setting: int,
    ) -> "LegacyPcrDataset":
        text_path = _as_path(path)
        columns = load_pcr_file(text_path)
        cleaned = clean_data(columns, integrations_per_setting=integrations_per_setting)
        return cls(
            path=text_path,
            columns=columns,
            cleaned=cleaned,
            integrations_per_setting=integrations_per_setting,
        )

    def threshold_values(self) -> np.ndarray:
        return np.asarray(self.cleaned["thresholds_TS"], dtype=float)

    def _resolve_threshold_index(self, selector: int | float | str | None) -> int:
        values = self.threshold_values()
        if values.size == 0:
            raise ValueError(f"No thresholds found in {self.path}")
        if selector is None or selector == "auto":
            best_index = None
            best_peak = -np.inf
            for idx in range(1, len(values) + 1):
                counts = np.asarray(self.cleaned[f"counts_TS{idx}"], dtype=float)
                delta_t = np.asarray(self.cleaned[f"delta_time_TS{idx}"], dtype=float)
                peak = np.nanmax(np.divide(counts, delta_t, where=delta_t > 0))
                if peak > best_peak:
                    best_peak = peak
                    best_index = idx
            return int(best_index)
        if isinstance(selector, str):
            stripped = selector.strip()
            if stripped.upper().startswith("TS"):
                selector = int(stripped[2:])
            else:
                selector = float(stripped)
        if isinstance(selector, int):
            if 1 <= selector <= len(values):
                return selector
            raise KeyError(f"Threshold series TS{selector} not found in {self.path}")
        selector_value = float(selector)
        nearest = int(np.argmin(np.abs(values - selector_value))) + 1
        return nearest

    def get_curve(
        self,
        selector: int | float | str | None = None,
        *,
        label: str | None = None,
    ) -> PcrCurve:
        idx = self._resolve_threshold_index(selector)
        counts = _as_array(self.cleaned[f"counts_TS{idx}"])
        delta_t = _as_array(self.cleaned[f"delta_time_TS{idx}"])
        bias = _as_array(self.cleaned[f"bias_TS{idx}"])
        count_rate = np.divide(counts, delta_t, where=delta_t > 0)
        count_rate_error = np.divide(
            np.sqrt(np.clip(counts, a_min=0, a_max=None)),
            delta_t,
            where=delta_t > 0,
        )
        threshold_value = float(self.threshold_values()[idx - 1])
        return PcrCurve(
            bias_current=np.abs(bias),
            counts=count_rate,
            counts_error=count_rate_error,
            threshold=ThresholdOption(index=idx, value=threshold_value),
            metadata={
                "integrations_per_setting": self.integrations_per_setting,
                "source_format": "legacy_text",
            },
            source_path=str(self.path),
            label=label,
        )


class _LegacyPickleCarrier:
    """Compatibility container for historic PCR pickle objects."""


class _LegacyPcrUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if name == "pcrData_varThresh":
            return _LegacyPickleCarrier
        return super().find_class(module, name)


def load_pickled_pcr_object(path: str | Path) -> Any:
    with _as_path(path).open("rb") as handle:
        return _LegacyPcrUnpickler(handle).load()


def load_pickled_pcr_curve(path: str | Path, *, label: str | None = None) -> PcrCurve:
    pickle_path = _as_path(path)
    data = load_pickled_pcr_object(pickle_path)

    bias_current = np.asarray(data.biasCurrent, dtype=float)
    _, first_indices = np.unique(bias_current, return_index=True)
    ordered_indices = np.sort(first_indices)
    unique_bias = bias_current[ordered_indices]

    counts_optimal = np.zeros(len(unique_bias), dtype=float)
    counts_exposure = np.zeros(len(unique_bias), dtype=float)
    dark_optimal = np.zeros(len(unique_bias), dtype=float)
    dark_error = np.zeros(len(unique_bias), dtype=float)
    times = np.full(len(unique_bias), np.nan, dtype=float)

    for idx, bias in enumerate(unique_bias):
        mask = (
            (data.biasCurrent == bias)
            * (data.thresh >= data.bestThreshes[idx, 0])
            * (data.thresh <= data.bestThreshes[idx, 1])
        )
        counts_optimal[idx] = np.sum(data.light_counts[mask])
        counts_exposure[idx] = np.sum(mask) * data.expTime * data.light_counts.shape[2]
        dark_optimal[idx] = np.sum(data.dark_counts[mask])
        dark_error[idx] = np.sqrt(dark_optimal[idx])
        if hasattr(data, "times"):
            selected_times = np.asarray(data.times)[mask]
            if selected_times.size:
                times[idx] = float(np.mean(selected_times))

    exposure_ratio = data.light_counts.shape[-1] / data.dark_counts.shape[-1]
    counts_rate = (counts_optimal - dark_optimal * exposure_ratio) / counts_exposure
    counts_error = (
        np.sqrt(counts_optimal + dark_optimal * exposure_ratio**2) / counts_exposure
    )
    dark_rate = dark_optimal * exposure_ratio / counts_exposure
    dark_rate_error = dark_error * exposure_ratio / counts_exposure

    if np.all(unique_bias <= 1e-9):
        unique_bias = np.abs(unique_bias)

    return PcrCurve(
        bias_current=unique_bias,
        counts=counts_rate,
        dark_counts=dark_rate,
        counts_error=counts_error,
        dark_counts_error=dark_rate_error,
        metadata={
            "source_format": "legacy_pickle",
            "times": times,
            "best_thresholds": np.asarray(data.bestThreshes, dtype=float),
        },
        source_path=str(pickle_path),
        label=label,
    )


def load_hero_pcr_csv(path: str | Path) -> ThresholdSweepDataset:
    return ThresholdSweepDataset.from_csv(path)


def load_legacy_pcr_text(
    path: str | Path,
    *,
    integrations_per_setting: int,
) -> LegacyPcrDataset:
    return LegacyPcrDataset.from_text(
        path,
        integrations_per_setting=integrations_per_setting,
    )


def load_legacy_light_dark_curve(
    light_path: str | Path,
    dark_path: str | Path,
    *,
    integrations_per_setting: int,
    threshold: int | float | str | None = None,
    label: str | None = None,
) -> PcrCurve:
    light = load_legacy_pcr_text(
        light_path,
        integrations_per_setting=integrations_per_setting,
    )
    dark = load_legacy_pcr_text(
        dark_path,
        integrations_per_setting=integrations_per_setting,
    )
    light_curve = light.get_curve(threshold, label=label)
    dark_curve = dark.get_curve(threshold, label=label)
    light_curve.dark_counts = dark_curve.counts
    light_curve.dark_counts_error = dark_curve.counts_error
    light_curve.metadata.update(
        {
            "dark_path": str(_as_path(dark_path)),
            "dark_threshold": (
                dark_curve.threshold.value if dark_curve.threshold else None
            ),
        }
    )
    return light_curve


class PcrLoader:
    """Convenience facade for the three PCR data formats used in this workspace."""

    @staticmethod
    def from_threshold_csv(path: str | Path) -> ThresholdSweepDataset:
        return load_hero_pcr_csv(path)

    @staticmethod
    def from_legacy_text(
        path: str | Path,
        *,
        integrations_per_setting: int,
    ) -> LegacyPcrDataset:
        return load_legacy_pcr_text(
            path,
            integrations_per_setting=integrations_per_setting,
        )

    @staticmethod
    def from_legacy_pair(
        light_path: str | Path,
        dark_path: str | Path,
        *,
        integrations_per_setting: int,
        threshold: int | float | str | None = None,
        label: str | None = None,
    ) -> PcrCurve:
        return load_legacy_light_dark_curve(
            light_path,
            dark_path,
            integrations_per_setting=integrations_per_setting,
            threshold=threshold,
            label=label,
        )

    @staticmethod
    def from_pickle(path: str | Path, *, label: str | None = None) -> PcrCurve:
        return load_pickled_pcr_curve(path, label=label)
