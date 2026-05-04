"""Generate matplotlib plots and metadata JSON for the 4.23.2026 hero data
presentation.

Outputs (written to both `mir_paper/out/4.23.2026/` and
`presentation/static/plots/`):

  power_ramp_<dev>um.png   — POWERSCAN CSV with linear fit overlay
  hist_<dev>um.png         — HIST JSON overlay
  pcr_<dev>um.png          — latest PCR CSV (3 trigger levels)
  pcr_combined.png         — 9 PCR curves combined
  metadata_<dev>um.json    — parsed filename metadata grouped by header

`<dev>` is the *displayed* size (46, 63, 87). The 87 µm device lives in the
`80um/` folder (mislabeled).

PCR x-axis uses the raw `Bias_Current` column unscaled (already in µA).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from util.pcr_loader import PcrLoader  # noqa: E402

try:
    from snsphd.viz import phd_style

    phd_style(jupyterStyle=False)
except Exception:  # pragma: no cover — optional dependency
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25})


PROJECT_ROOT = SRC_DIR.parents[1]
DATA_ROOT = (
    SRC_DIR.parent / "data" / "DC_Pa250528a_60nm_hero" / "4.23.2026"
)
OUT_DIR = SRC_DIR.parent / "out" / "4.23.2026"
PRESENTATION_PLOTS = PROJECT_ROOT / "presentation" / "static" / "plots"

DEVICES = [
    {
        "folder": "46um",
        "label": "46 µm",
        "display": "46",
        "fit_range_mA": (240, 280),
        "power_ylim": None,
        "hist_ylim": (0, 0.3e6),
        "pcr_xlim": (0, 0.095),
        "pcr_ylim": (0, 100000),
    },
    {
        "folder": "63um",
        "label": "63 µm",
        "display": "63",
        "fit_range_mA": (35, 64),
        "power_ylim": (0, 75000),
        "hist_ylim": (0, 75000),
        "pcr_xlim": (0, 0.095),
        "pcr_ylim": (0, 70000),
    },
    {
        "folder": "80um",
        "label": "87 µm",
        "display": "87",
        "fit_range_mA": (60, 100),
        "power_ylim": (0, 125000),
        "hist_ylim": None,
        "pcr_xlim": (0, 0.095),
        "pcr_ylim": (0, 75000),
    },
]

DEVICE_COLORS = {
    "46": plt.cm.plasma(0.38),
    "63": plt.cm.plasma(0.55),
    "87": plt.cm.plasma(0.85),
}

TL_LINESTYLES = ["-", "--", ":"]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def latest_pcr_csv(folder: Path) -> Path:
    candidates = [
        p
        for p in folder.glob("PCR__*.csv")
        if not p.name.startswith("NEGATIVE_PCR")
        and not p.name.startswith("test_PCR")
    ]
    if not candidates:
        raise FileNotFoundError(f"No PCR CSV in {folder}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_powerscan_csv(folder: Path) -> Path:
    candidates = list(folder.glob("POWERSCAN*.csv"))
    if not candidates:
        candidates = [
            p for p in folder.glob("power_ramp*.csv") if "test" not in p.name
        ]
    if not candidates:
        raise FileNotFoundError(f"No power scan CSV in {folder}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_power_ramp_csv(path: Path) -> pd.DataFrame:
    with path.open() as fh:
        lines = fh.readlines()
    skip = next(
        i for i, line in enumerate(lines) if line.startswith("QCL_Current_mA")
    )
    return pd.read_csv(path, skiprows=skip)


# ---------------------------------------------------------------------------
# Filename parsers (use the supplied templates; tolerant of stray brackets and
# unit suffixes like "us"/"Hz" embedded in raw values).
# ---------------------------------------------------------------------------


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def numpart(s: str) -> str:
    m = _NUM_RE.search(s.strip().rstrip("]"))
    return m.group(0) if m else s.strip()


def _entry(label: str, raw: str, unit: str) -> dict:
    return {"label": label, "value": f"{numpart(raw)} {unit}".strip()}


def parse_hist_filename(name: str) -> dict:
    """HIST__QCL_w_V_I_ontime_reprate__FRIDGE_temp__SAVE_inttime_trigger_DETECTOR_bias"""
    stem = Path(name).stem
    parts = stem.split("__")
    qv = parts[1].split("_")[1:]
    fv = parts[2].split("_")[1:]
    save_part, _, det_part = parts[3].partition("_DETECTOR_")
    sv = save_part.split("_")[1:]
    return {
        "QCL": [
            _entry("wavelength", qv[0], "µm"),
            _entry("voltage", qv[1], "V"),
            _entry("current", qv[2], "mA"),
            _entry("ontime", qv[3], "µs"),
            _entry("reprate", qv[4], "Hz"),
        ],
        "FRIDGE": [_entry("temp", fv[0], "mK")],
        "SAVE": [
            _entry("inttime", sv[0], "s"),
            _entry("trigger", sv[1], "mV"),
        ],
        "DETECTOR": [_entry("bias", det_part, "µA")],
    }


def parse_powerscan_filename(name: str) -> dict:
    """POWERSCAN_QCL_w_ontime_reprate__FRIDGE_temp__SAVE_inttime_trigger_DETECTOR_bias__WINDOWT_onstart_onstop_..."""
    stem = Path(name).stem
    parts = stem.split("__")
    qv = parts[0].replace("POWERSCAN_QCL_", "").split("_")
    fv = parts[1].split("_")[1:]
    save_part, _, det_part = parts[2].partition("_DETECTOR_")
    sv = save_part.split("_")[1:]
    wv = parts[3].replace("WINDOWT_", "").split("_")
    return {
        "QCL": [
            _entry("wavelength", qv[0], "µm"),
            _entry("ontime", qv[1], "µs"),
            _entry("reprate", qv[2], "Hz"),
        ],
        "FRIDGE": [_entry("temp", fv[0], "mK")],
        "SAVE": [
            _entry("inttime", sv[0], "s"),
            _entry("trigger", sv[1], "mV"),
        ],
        "DETECTOR": [_entry("bias", det_part, "µA")],
        "WINDOWT": [
            _entry("onstart", wv[0], "ms"),
            _entry("onstop", wv[1], "ms"),
        ],
    }


def parse_pcr_filename(name: str) -> dict:
    """PCR__QCL_w_V_I_ontime_reprate__WINDOWT_<6 vals>__FRIDGE_temp_SAVET_inttime_(cycles_)?val"""
    stem = Path(name).stem
    parts = stem.split("__")
    qv = parts[1].split("_")[1:]
    wv = parts[2].split("_")[1:]
    last = parts[3].split("_")
    # last is e.g. ['FRIDGE', '259', 'SAVET', '4', 'cycles', '8']
    # or         ['FRIDGE', '264', 'SAVET', '2', '6']
    fridge_temp = last[last.index("FRIDGE") + 1]
    savet_idx = last.index("SAVET")
    savet_inttime = last[savet_idx + 1]
    cycles_val = last[-1] if last[-1] != savet_inttime else last[savet_idx + 2]
    return {
        "QCL": [
            _entry("wavelength", qv[0], "µm"),
            _entry("voltage", qv[1], "V"),
            _entry("current", qv[2], "mA"),
            _entry("ontime", qv[3], "µs"),
            _entry("reprate", qv[4], "Hz"),
        ],
        "WINDOWT": [
            _entry("onstart", wv[0], "ms"),
            _entry("onstop", wv[1], "ms"),
            _entry("ondcrstart", wv[2], "ms"),
            _entry("ondcrstop", wv[3], "µs"),
            _entry("offstart", wv[4], "ms"),
            _entry("offstop", wv[5], "ms"),
        ],
        "FRIDGE": [_entry("temp", fridge_temp, "mK")],
        "SAVET": [
            _entry("inttime", savet_inttime, "s"),
            _entry("cycles", cycles_val, ""),
        ],
    }


def hist_overlay_metadata(hist_files: list[Path]) -> dict | None:
    """Collapse the per-file voltage/current into a range across the overlay."""
    if not hist_files:
        return None
    first = parse_hist_filename(hist_files[0].name)
    last = parse_hist_filename(hist_files[-1].name)
    by_label_first = {item["label"]: item["value"] for item in first["QCL"]}
    by_label_last = {item["label"]: item["value"] for item in last["QCL"]}

    def rng(label: str) -> str:
        a = by_label_first[label]
        b = by_label_last[label]
        return a if a == b else f"{a} – {b}"

    return {
        "QCL": [
            {"label": "wavelength", "value": by_label_first["wavelength"]},
            {"label": "voltage", "value": rng("voltage")},
            {"label": "current", "value": rng("current")},
            {"label": "ontime", "value": by_label_first["ontime"]},
            {"label": "reprate", "value": by_label_first["reprate"]},
        ],
        "FRIDGE": first["FRIDGE"],
        "SAVE": first["SAVE"],
        "DETECTOR": first["DETECTOR"],
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRESENTATION_PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=200, bbox_inches="tight")
    fig.savefig(PRESENTATION_PLOTS / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_power_ramp(folder: Path, dev: dict, csv_path: Path) -> None:
    df = load_power_ramp_csv(csv_path)
    color = DEVICE_COLORS[dev["display"]]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(
        df["QCL_Current_mA"],
        df["Signal_Hz"],
        marker="o",
        markersize=3,
        linewidth=1.6,
        color=color,
        label="Data",
    )
    ax.errorbar(
        df["QCL_Current_mA"],
        df["Signal_Hz"],
        yerr=10 * np.sqrt(np.abs(df["Signal_Hz"])),
        xerr=1.5,
        capsize=0,
        alpha=0.3,
        color=color,
        elinewidth=4,
    )

    fit_lo, fit_hi = dev["fit_range_mA"]
    mask = (df["QCL_Current_mA"] >= fit_lo) & (df["QCL_Current_mA"] <= fit_hi)
    fit_x = df.loc[mask, "QCL_Current_mA"].to_numpy()
    fit_y = df.loc[mask, "Signal_Hz"].to_numpy()
    if fit_x.size >= 2:
        slope, intercept = np.polyfit(fit_x, fit_y, deg=1)
        xs = np.linspace(fit_lo, fit_hi, 50)
        ax.plot(
            xs,
            slope * xs + intercept,
            color="black",
            linewidth=2.2,
            linestyle="--",
            label=(
                f"Linear fit ({fit_lo}–{fit_hi} mA): "
                f"slope = {slope:,.1f} Hz/mA"
            ),
        )

    if dev["power_ylim"] is not None:
        ax.set_ylim(*dev["power_ylim"])
    ax.set_xlabel("QCL Current (mA)")
    ax.set_ylabel("SNSPD Count Rate (Hz)")
    ax.set_title(f"{dev['label']} — QCL Power Ramp")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="best")
    save(fig, f"power_ramp_{dev['display']}um.png")


def _hist_label(filename: str) -> str:
    m = re.search(r"QCL_\d+_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)", filename)
    if m:
        return f"{m.group(1)} V, {m.group(2)} mA"
    return filename


def plot_hist(dev: dict, hist_files: list[Path], downsample: int = 4) -> None:
    if not hist_files:
        return
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for i, path in enumerate(hist_files):
        color = plt.cm.viridis(i / max(len(hist_files) - 1, 1))
        with path.open() as fh:
            payload = json.load(fh)
        x_us = np.asarray(payload["x_axis_ps"], dtype=float) / 1e6
        y_rate = np.asarray(
            payload["instantaneous_count_rate_hz"], dtype=float
        )
        n = (len(x_us) // downsample) * downsample
        if n == 0:
            continue
        x_us = x_us[:n].reshape(-1, downsample).mean(axis=1)
        y_rate = y_rate[:n].reshape(-1, downsample).mean(axis=1)
        ax.plot(
            x_us,
            y_rate,
            linewidth=1.4,
            alpha=0.92,
            color=color,
            label=_hist_label(path.name),
        )
    if dev["hist_ylim"] is not None:
        ax.set_ylim(*dev["hist_ylim"])
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Instantaneous Count Rate (cps)")
    ax.set_title(f"{dev['label']} — HIST Overlay")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="best")
    save(fig, f"hist_{dev['display']}um.png")


def plot_pcr_single(dev: dict, csv_path: Path) -> None:
    dataset = PcrLoader.from_threshold_csv(csv_path)
    options = dataset.threshold_options()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cmap = plt.cm.plasma
    for i, opt in enumerate(options):
        curve = dataset.get_curve(opt.index)
        color = cmap(0.15 + 0.7 * (i / max(len(options) - 1, 1)))
        ax.plot(
            curve.bias_current,  # raw bias — already in µA per data convention
            curve.counts,
            marker="o",
            markersize=3,
            linewidth=1.6,
            color=color,
            label=f"TL{opt.index} ({opt.value:.3f} V)",
        )
    ax.set_xlim(*dev["pcr_xlim"])
    ax.set_ylim(*dev["pcr_ylim"])
    ax.set_xlabel(r"Bias Current ($\mu$A)")
    ax.set_ylabel("Photon Count Rate (cps)")
    ax.set_title(f"{dev['label']} — PCR")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="best")
    save(fig, f"pcr_{dev['display']}um.png")


def plot_pcr_combined(name: str, norm_bias: float = 0.094) -> None:
    """Normalize each device so the average of its 3 trigger levels at
    `norm_bias` (or the nearest measured bias) equals 1.0."""
    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    for dev in DEVICES:
        folder = DATA_ROOT / dev["folder"]
        csv_path = latest_pcr_csv(folder)
        dataset = PcrLoader.from_threshold_csv(csv_path)
        color = DEVICE_COLORS[dev["display"]]
        options = dataset.threshold_options()
        curves = [dataset.get_curve(opt.index) for opt in options]

        bias = curves[0].bias_current
        idx = int(np.argmin(np.abs(bias - norm_bias)))
        norm_value = float(np.mean([c.counts[idx] for c in curves]))
        if not np.isfinite(norm_value) or norm_value <= 0:
            norm_value = 1.0

        for i, (opt, curve) in enumerate(zip(options, curves)):
            ls = TL_LINESTYLES[i % len(TL_LINESTYLES)]
            ax.plot(
                curve.bias_current,
                curve.counts / norm_value,
                color=color,
                linestyle=ls,
                linewidth=1.7,
                marker="o",
                markersize=3,
                label=f"{dev['label']} | TL{opt.index} ({opt.value:.3f})",
            )
    ax.set_xlim(0.025, 0.098)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"Bias Current ($\mu$A)")
    ax.set_ylabel("Normalized Photon Count Rate")
    ax.set_title(
        f"PCR — All QCL devices (normalized to 1.0 at {norm_bias:.3f} µA)"
    )
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    save(fig, name)


# ---------------------------------------------------------------------------
# Metadata serialization
# ---------------------------------------------------------------------------


def write_metadata(name: str, payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRESENTATION_PLOTS.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    (OUT_DIR / name).write_text(text, encoding="utf-8")
    (PRESENTATION_PLOTS / name).write_text(text, encoding="utf-8")


def main() -> None:
    for dev in DEVICES:
        folder = DATA_ROOT / dev["folder"]
        d = dev["display"]

        powerscan_path = latest_powerscan_csv(folder)
        hist_files = sorted(folder.glob("HIST__*.json"))
        pcr_path = latest_pcr_csv(folder)

        plot_power_ramp(folder, dev, powerscan_path)
        plot_hist(dev, hist_files)
        plot_pcr_single(dev, pcr_path)

        metadata = {
            "device_label": dev["label"],
            "power_ramp": {
                "filename": powerscan_path.name,
                "groups": parse_powerscan_filename(powerscan_path.name),
            },
            "hist": {
                "n_files": len(hist_files),
                "filename_pattern": (
                    hist_files[0].name if hist_files else None
                ),
                "groups": hist_overlay_metadata(hist_files) or {},
            },
            "pcr": {
                "filename": pcr_path.name,
                "groups": parse_pcr_filename(pcr_path.name),
            },
        }
        write_metadata(f"metadata_{d}um.json", metadata)

    plot_pcr_combined("pcr_combined.png")
    print(f"wrote plots and metadata to {OUT_DIR} and {PRESENTATION_PLOTS}")


if __name__ == "__main__":
    main()
