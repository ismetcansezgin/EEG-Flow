"""
tests/test_filters.py
=====================
Unit tests for the EEGFlow Digital Filtering Module (backend/utils/filters.py).

Tests verify:
  1. Filter design produces a valid SOS array.
  2. A single-channel band-pass filter correctly attenuates out-of-band noise.
  3. The filtered signal has the same length as the input.
  4. The DataFrame-level filter preserves non-channel columns.
  5. Invalid parameter combinations raise appropriate errors.

Run with:  python -m pytest tests/test_filters.py -v
"""

import numpy as np
import pandas as pd
import pytest

from backend.utils.filters import (
    design_bandpass_filter,
    apply_bandpass_filter,
    apply_filter_to_dataframe,
)

# ─── Test constants ────────────────────────────────────────
FS       = 250.0   # Sampling frequency (Hz)
LOWCUT   = 0.5     # High-pass cut-off (Hz)
HIGHCUT  = 45.0    # Low-pass  cut-off (Hz)
DURATION = 10      # Signal duration (seconds)
N        = int(FS * DURATION)  # Total samples = 2500


# ─── Helpers ───────────────────────────────────────────────
def make_sine(frequency: float, n: int = N, fs: float = FS) -> np.ndarray:
    """Generate a pure sine wave at the given frequency."""
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * frequency * t)


# ══════════════════════════════════════════════════════════
# TEST 1: Filter design — SOS array shape is correct
# ══════════════════════════════════════════════════════════
def test_design_bandpass_filter_returns_valid_sos():
    """design_bandpass_filter() must return a 2D SOS array."""
    sos = design_bandpass_filter(LOWCUT, HIGHCUT, FS, order=4)
    assert isinstance(sos, np.ndarray), "SOS must be a numpy array."
    assert sos.ndim == 2,               "SOS must be a 2D array (n_sections x 6)."
    assert sos.shape[1] == 6,           "Each SOS section must have exactly 6 coefficients."


# ══════════════════════════════════════════════════════════
# TEST 2: In-band signal passes through with high power
# ══════════════════════════════════════════════════════════
def test_inband_signal_passes_through():
    """
    A 10 Hz sine wave (inside the 0.5–45 Hz pass-band) should retain
    most of its amplitude after filtering.
    """
    signal   = make_sine(10.0)   # 10 Hz — inside the pass-band
    filtered = apply_bandpass_filter(signal, LOWCUT, HIGHCUT, FS)

    power_raw      = np.mean(signal ** 2)
    power_filtered = np.mean(filtered ** 2)

    # Filtered power should be at least 90% of raw power
    assert power_filtered > 0.9 * power_raw, (
        f"In-band signal lost too much power: "
        f"raw={power_raw:.4f}, filtered={power_filtered:.4f}"
    )


# ══════════════════════════════════════════════════════════
# TEST 3: Out-of-band signal (60 Hz noise) is strongly attenuated
# ══════════════════════════════════════════════════════════
def test_outofband_signal_is_attenuated():
    """
    A 60 Hz sine wave (above the 45 Hz cut-off) should be strongly
    attenuated (reduced to < 5% of its original power).
    """
    signal   = make_sine(60.0)   # 60 Hz — outside the pass-band
    filtered = apply_bandpass_filter(signal, LOWCUT, HIGHCUT, FS)

    power_raw      = np.mean(signal ** 2)
    power_filtered = np.mean(filtered ** 2)

    # Filtered power must be less than 5% of original
    assert power_filtered < 0.05 * power_raw, (
        f"Out-of-band signal (60 Hz) was not attenuated enough: "
        f"raw={power_raw:.4f}, filtered={power_filtered:.4f}"
    )


# ══════════════════════════════════════════════════════════
# TEST 4: Output length equals input length
# ══════════════════════════════════════════════════════════
def test_filtered_signal_length_unchanged():
    """The filtered signal must have the exact same number of samples as the input."""
    signal   = make_sine(10.0)
    filtered = apply_bandpass_filter(signal, LOWCUT, HIGHCUT, FS)
    assert len(filtered) == len(signal), (
        f"Length mismatch: input={len(signal)}, output={len(filtered)}"
    )


# ══════════════════════════════════════════════════════════
# TEST 5: DataFrame filter preserves non-EEG columns
# ══════════════════════════════════════════════════════════
def test_apply_filter_to_dataframe_preserves_metadata():
    """
    Non-EEG columns (time_ms, subject_id, event) must be identical
    before and after filtering.
    """
    t            = np.arange(N) / FS * 1000      # time in ms
    subject_ids  = np.ones(N, dtype=int)
    events       = np.array(["Relaxed"] * N)

    df = pd.DataFrame({
        "time_ms":    t,
        "subject_id": subject_ids,
        "event":      events,
        "Fp1":        make_sine(10.0) + make_sine(60.0),
        "Fp2":        make_sine(8.0)  + make_sine(60.0),
    })

    channels    = ["Fp1", "Fp2"]
    df_filtered = apply_filter_to_dataframe(df, channels, LOWCUT, HIGHCUT, FS)

    # Metadata columns must be unchanged
    pd.testing.assert_series_equal(df["time_ms"],    df_filtered["time_ms"])
    pd.testing.assert_series_equal(df["subject_id"], df_filtered["subject_id"])
    pd.testing.assert_series_equal(df["event"],      df_filtered["event"])

    # EEG columns must have been modified (filtered ≠ raw)
    assert not np.allclose(df["Fp1"].values, df_filtered["Fp1"].values), \
        "Fp1 channel was not modified by filtering."


# ══════════════════════════════════════════════════════════
# TEST 6: Invalid parameters raise ValueError
# ══════════════════════════════════════════════════════════
def test_invalid_lowcut_greater_than_highcut_raises():
    """lowcut >= highcut must raise ValueError."""
    with pytest.raises(ValueError, match="must be less than highcut"):
        design_bandpass_filter(lowcut=50.0, highcut=10.0, fs=FS)


def test_highcut_above_nyquist_raises():
    """highcut >= Nyquist frequency must raise ValueError."""
    with pytest.raises(ValueError, match="Nyquist"):
        design_bandpass_filter(lowcut=0.5, highcut=130.0, fs=FS)  # Nyquist = 125 Hz


def test_signal_too_short_raises():
    """Signals shorter than 3*(2*order) samples must raise ValueError."""
    too_short = np.ones(5)  # Way below minimum
    with pytest.raises(ValueError, match="Signal too short"):
        apply_bandpass_filter(too_short, LOWCUT, HIGHCUT, FS)
