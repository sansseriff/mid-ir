from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Any

import numpy as np


def _clean_col_name(raw: str) -> str:
    """Convert a verbose column name (with units) to a pythonic snake_case key.

    Examples
    - "Bias-[V]" -> "bias_V"
    - "Counts-[#Photons]" -> "counts"
    - "Min Peak Voltage-[mV]" -> "min_peak_voltage_mV"
    """
    # Extract unit inside square brackets if present
    unit_match = re.search(r"\[(.*?)\]", raw)
    unit = unit_match.group(1) if unit_match else ""
    # Normalize unit: keep simple alphanumerics only (drop symbols like '#')
    unit_norm = re.sub(r"[^0-9a-zA-Z]+", "", unit)

    # Remove unit portion and any trailing hyphen before it
    base = re.sub(r"-?\s*\[.*?\]", "", raw)
    # Normalize base to snake_case
    base = base.strip().lower()
    base = re.sub(r"[^0-9a-zA-Z]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")

    if unit_norm:
        return f"{base}_{unit_norm}"
    return base


def _parse_header_for_columns(text: str) -> list[str] | None:
    """Find the '# Cols:' line and return the list of raw column names.

    The header line is expected to look like:
    '# Cols: unixtime-[s]\t Bias-[V]\t Sense_Voltage-[V]\t ...'
    We'll first try splitting by tabs, then fallback to splitting on 2+ spaces
    to handle files that don't use tabs.
    """
    cols_line = None
    for line in text.splitlines():
        if line.lstrip().startswith("#") and "Cols:" in line:
            cols_line = line
            break
    if not cols_line:
        return None

    # Take the part after 'Cols:'
    after = cols_line.split("Cols:", 1)[1].strip()
    if "\t" in after:
        parts = [p.strip() for p in after.split("\t") if p.strip()]
    else:
        # Split on 2+ spaces to avoid breaking names that contain single spaces
        parts = [p.strip() for p in re.split(r"\s{2,}", after) if p.strip()]

    return parts if parts else None


def load_pcr_file(path: str | Path) -> Dict[str, Any]:
    """Load a 'Photon Count Rate vs Bias' text file into column arrays.

    Input
    - path: str | Path to the ASCII data file. The file contains comment lines
      beginning with '#', and a header line of the form '# Cols: ...'.

    Output
    - A dict mapping cleaned, pythonic column names to numpy arrays. Two
      convenience aliases are added when available:
            - 'bias' -> the Bias column (in volts)
            - 'counts' -> the Counts column

    Notes
    - All columns are returned; names include a unit suffix when available
      (e.g., bias_V, threshold_mV).
    - If the '# Cols:' header is missing, generic names like col0, col1, ...
      are used.

    Example
    >>> arrays = load_pcr_file("/path/to/pcrcurve.txt")
    >>> bias = arrays["bias"]
    >>> counts = arrays["counts"]
    >>> # All columns are available via their cleaned names
    >>> arrays.keys()  # e.g., dict_keys(['unixtime_s', 'bias_V', 'sense_voltage_V', ...])
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")

    # Discover raw column names from the '# Cols:' line, if present
    raw_cols = _parse_header_for_columns(text) or []

    # Prepare names for pandas: if none found, we'll infer count from first data row
    # Keep raw column names if found; otherwise we'll infer later

    # Read numeric block with numpy, skipping comment lines
    data = np.genfromtxt(str(p), comments="#", dtype=float)
    if data.ndim == 1:
        # Single-row edge case
        data = data.reshape(1, -1)

    # If we didn't have header names, create generic ones based on column count
    if not raw_cols:
        raw_cols = [f"col{i}" for i in range(data.shape[1])]

    # Build output dict with cleaned names from numpy columns
    out: Dict[str, Any] = {}
    name_map: dict[str, str] = {}
    for idx, raw in enumerate(raw_cols):
        clean = _clean_col_name(str(raw))
        # Ensure uniqueness if two different raw names clean to same key
        base = clean
        suffix = 1
        while clean in out:
            suffix += 1
            clean = f"{base}_{suffix}"
        out[clean] = data[:, idx]
        name_map[raw] = clean

    # Convenience aliases commonly used by downstream plotting
    # Bias
    for candidate in ("Bias-[V]", "Bias", "bias", "bias_V"):
        if candidate in name_map:
            out.setdefault("bias", out[name_map[candidate]])
            break
        if candidate in out:
            out.setdefault("bias", out[candidate])
            break

    # Counts
    for candidate in ("Counts-[#Photons]", "Counts", "counts"):
        if candidate in name_map:
            out.setdefault("counts", out[name_map[candidate]])
            break
        if candidate in out:
            out.setdefault("counts", out[candidate])
            break

    return out


def clean_data(
    columns: Dict[str, Any], integrations_per_setting: int
) -> Dict[str, Any]:
    """General data cleaning and coalescing for PCR files.

     Steps performed:
     1. Coalesce data in fixed-size blocks of length `integrations_per_setting`
         (n): assume rows i..i+n-1 share the same settings except counts and
         possibly small analog fluctuations. For each block, sum counts and set
         delta_time = last_unixtime - first_unixtime.
     2. After coalescing, bifurcate the coalesced arrays by unique threshold
       values producing per-threshold arrays named like 'counts_TS1',
       'bias_TS1', 'delta_time_TS1', 'min_peak_voltage_TS1',
       'peak_to_peak_TS1', etc.

    Input
        - columns: dict returned by load_pcr_file
        - integrations_per_setting: int, number of repeated integrations per
            settings block (e.g., 5)

    Output
    - A dict containing the original cleaned columns plus coalesced arrays
      and per-threshold arrays. Important new keys:
        - 'coalesced_counts', 'coalesced_bias', 'coalesced_delta_time',
          'coalesced_thresholds', 'coalesced_min_peak_voltage_mV',
          'coalesced_peak_to_peak_voltage_mV'
        - 'counts_TS1'..'counts_TSN', 'bias_TS1'.., 'delta_time_TS1'..,
          'min_peak_voltage_TS1'.., 'peak_to_peak_TS1'..
        - 'thresholds_TS' (numpy array of TS values in order)
                - Convenience duplicates for shared per-TS arrays: 'bias_TS',
                    'delta_time_TS', 'min_peak_voltage_TS', 'peak_to_peak_TS' that copy
                    the TS1 arrays (these arrays should be identical across thresholds).

    The function attempts to find the appropriate keys automatically. It will
    raise ValueError if required columns (counts, unixtime, threshold, bias)
    cannot be located.
    """
    out = dict(columns)

    if integrations_per_setting <= 0:
        raise ValueError("integrations_per_setting must be a positive integer")

    # Helper to find a key by predicate
    def _find_key(prefixes: list[str]) -> str | None:
        for p in prefixes:
            for k in out:
                if k.lower().startswith(p.lower()):
                    return k
        return None

    counts_key = _find_key(["counts"]) or None
    if counts_key is None:
        raise ValueError("counts column not found in input dict")

    unixtime_key = _find_key(["unixtime"]) or None
    if unixtime_key is None:
        raise ValueError("unixtime column not found in input dict")

    # threshold, bias, min/max peak, ptp
    threshold_key = _find_key(["threshold"]) or None
    bias_key = _find_key(["bias"]) or None
    min_peak_key = None
    ptp_key = None
    # try more targeted matches
    for k in out:
        kl = k.lower()
        if "min" in kl and "peak" in kl:
            min_peak_key = k
        if "peak_to_peak" in kl or "peaktopeak" in kl or "ptp" in kl:
            ptp_key = k

    if threshold_key is None:
        raise ValueError("threshold column not found in input dict")
    if bias_key is None:
        raise ValueError("bias column not found in input dict")
    # min_peak_key and ptp_key are optional but we'll continue without them

    counts_arr = np.asarray(out[counts_key])
    time_arr = np.asarray(out[unixtime_key], dtype=float)

    nrows = len(counts_arr)
    if len(time_arr) != nrows:
        raise ValueError("counts and unixtime arrays must have the same length")

    # Coalesce fixed-size blocks
    co_counts: list[float] = []
    co_bias: list[float] = []
    co_thresholds: list[float] = []
    co_delta_time: list[float] = []
    co_min_peak: list[float] = []
    co_ptp: list[float] = []

    n = integrations_per_setting
    last_full = (nrows // n) * n
    for start in range(0, last_full, n):
        end = start + n  # exclusive
        idx0 = start
        idxn = end - 1
        co_counts.append(float(np.sum(counts_arr[start:end])))
        co_bias.append(float(np.asarray(out[bias_key][idx0])))
        co_thresholds.append(float(np.asarray(out[threshold_key][idx0])))
        co_delta_time.append(float(time_arr[idxn] - time_arr[idx0]))
        if min_peak_key and min_peak_key in out:
            co_min_peak.append(float(np.asarray(out[min_peak_key][idx0])))
        else:
            co_min_peak.append(np.nan)
        if ptp_key and ptp_key in out:
            co_ptp.append(float(np.asarray(out[ptp_key][idx0])))
        else:
            co_ptp.append(np.nan)

    co_counts_arr = np.array(co_counts)
    co_bias_arr = np.array(co_bias)
    co_thresholds_arr = np.array(co_thresholds)
    co_delta_time_arr = np.array(co_delta_time)
    co_min_peak_arr = np.array(co_min_peak)
    co_ptp_arr = np.array(co_ptp)

    # Store coalesced arrays
    out["coalesced_counts"] = co_counts_arr
    out["coalesced_bias"] = co_bias_arr
    out["coalesced_thresholds"] = co_thresholds_arr
    out["coalesced_delta_time"] = co_delta_time_arr
    out["coalesced_min_peak_voltage_mV"] = co_min_peak_arr
    out["coalesced_peak_to_peak_voltage_mV"] = co_ptp_arr

    # Now bifurcate by threshold values
    thr_values = sorted({float(v) for v in co_thresholds_arr.tolist()})
    out["thresholds_TS"] = np.array(thr_values, dtype=float)

    for i, thr in enumerate(thr_values, start=1):
        mask = co_thresholds_arr == thr
        out[f"counts_TS{i}"] = co_counts_arr[mask]
        out[f"bias_TS{i}"] = co_bias_arr[mask]
        out[f"delta_time_TS{i}"] = co_delta_time_arr[mask]
        out[f"min_peak_voltage_TS{i}"] = co_min_peak_arr[mask]
        out[f"peak_to_peak_voltage_TS{i}"] = co_ptp_arr[mask]

    # Convenience duplicates (shared across TS): copy TS1 versions
    if thr_values:
        out["bias_TS"] = out.get("bias_TS1")
        out["delta_time_TS"] = out.get("delta_time_TS1")
        out["min_peak_voltage_TS"] = out.get("min_peak_voltage_TS1")
        out["peak_to_peak_voltage_TS"] = out.get("peak_to_peak_voltage_TS1")

    return out


def _safe_value(v: Any) -> Any:
    """Return a hashable representation of a value for comparison in settings.

    Floats are converted to rounded floats (12 decimal places) to avoid tiny
    binary-representation differences causing unequal tuples. Other values are
    returned as-is.
    """
    try:
        if isinstance(v, (float, int, np.floating, np.integer)):
            return float(round(float(v), 12))
    except Exception:
        pass
    return v
