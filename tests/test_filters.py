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
    apply_notch_filter,
    apply_notch_to_dataframe,
    apply_linear_detrend,
    apply_detrend_to_dataframe,
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
    signal   = make_sine(10.0)
    filtered = apply_bandpass_filter(signal, LOWCUT, HIGHCUT, FS)

    power_raw      = np.mean(signal ** 2)
    power_filtered = np.mean(filtered ** 2)

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
    signal   = make_sine(60.0)
    filtered = apply_bandpass_filter(signal, LOWCUT, HIGHCUT, FS)

    power_raw      = np.mean(signal ** 2)
    power_filtered = np.mean(filtered ** 2)

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
    assert len(filtered) == len(signal)


# ══════════════════════════════════════════════════════════
# TEST 5: DataFrame filter preserves non-EEG columns
# ══════════════════════════════════════════════════════════
def test_apply_filter_to_dataframe_preserves_metadata():
    """
    Non-EEG columns (time_ms, subject_id, event) must be identical
    before and after filtering.
    """
    t = np.arange(N) / FS * 1000

    df = pd.DataFrame({
        "time_ms":    t,
        "subject_id": np.ones(N, dtype=int),
        "event":      np.array(["Relaxed"] * N),
        "Fp1":        make_sine(10.0) + make_sine(60.0),
        "Fp2":        make_sine(8.0)  + make_sine(60.0),
    })

    channels    = ["Fp1", "Fp2"]
    df_filtered = apply_filter_to_dataframe(df, channels, LOWCUT, HIGHCUT, FS)

    pd.testing.assert_series_equal(df["time_ms"],    df_filtered["time_ms"])
    pd.testing.assert_series_equal(df["subject_id"], df_filtered["subject_id"])
    pd.testing.assert_series_equal(df["event"],      df_filtered["event"])

    assert not np.allclose(df["Fp1"].values, df_filtered["Fp1"].values), \
        "Fp1 channel was not modified by filtering."


# ══════════════════════════════════════════════════════════
# TEST 6-8: Invalid parameters raise ValueError
# ══════════════════════════════════════════════════════════
def test_invalid_lowcut_greater_than_highcut_raises():
    """lowcut >= highcut must raise ValueError."""
    with pytest.raises(ValueError, match="must be less than highcut"):
        design_bandpass_filter(lowcut=50.0, highcut=10.0, fs=FS)


def test_highcut_above_nyquist_raises():
    """highcut >= Nyquist frequency must raise ValueError."""
    with pytest.raises(ValueError, match="Nyquist"):
        design_bandpass_filter(lowcut=0.5, highcut=130.0, fs=FS)


def test_signal_too_short_raises():
    """Signals shorter than 3*(2*order) samples must raise ValueError."""
    too_short = np.ones(5)
    with pytest.raises(ValueError, match="Signal too short"):
        apply_bandpass_filter(too_short, LOWCUT, HIGHCUT, FS)


# ════════════════════════════════════════════════════════
# NOTCH FILTER TESTS
# ════════════════════════════════════════════════════════

def test_notch_filter_attenuates_target_frequency():
    """
    A 50 Hz sine wave must be strongly attenuated (< 5% power remaining)
    after applying a 50 Hz Notch filter.
    """
    signal   = make_sine(50.0)
    filtered = apply_notch_filter(signal, notch_freq=50.0, fs=FS)

    power_raw      = np.mean(signal ** 2)
    power_filtered = np.mean(filtered ** 2)

    assert power_filtered < 0.05 * power_raw, (
        f"Notch filter did not attenuate 50 Hz. "
        f"raw={power_raw:.4f}, filtered={power_filtered:.4f}"
    )


def test_notch_filter_preserves_other_frequencies():
    """
    A 10 Hz sine wave (far from the 50 Hz notch) must retain > 95% of
    its power after notch filtering.
    """
    signal   = make_sine(10.0)
    filtered = apply_notch_filter(signal, notch_freq=50.0, fs=FS)

    power_raw      = np.mean(signal ** 2)
    power_filtered = np.mean(filtered ** 2)

    assert power_filtered > 0.95 * power_raw, (
        f"Notch filter damaged 10 Hz signal. "
        f"raw={power_raw:.4f}, filtered={power_filtered:.4f}"
    )


def test_notch_filter_output_length_unchanged():
    """Notch-filtered signal must have the same number of samples as input."""
    signal   = make_sine(10.0)
    filtered = apply_notch_filter(signal, notch_freq=50.0, fs=FS)
    assert len(filtered) == len(signal)


def test_notch_filter_invalid_frequency_raises():
    """A notch_freq above Nyquist must raise ValueError."""
    with pytest.raises(ValueError, match="Nyquist"):
        apply_notch_filter(make_sine(10.0), notch_freq=200.0, fs=FS)


# ════════════════════════════════════════════════════════
# LINEAR DETRENDING TESTS
# ════════════════════════════════════════════════════════

def test_linear_detrend_removes_known_slope():
    """
    A pure linear ramp (y = slope * t) should be reduced to near-zero
    after linear detrending. Residual RMS must be < 1% of original.
    """
    t         = np.arange(N) / FS
    ramp      = 5.0 * t                  # 5 µV/s upward drift, no oscillation
    detrended = apply_linear_detrend(ramp)

    rms_original  = np.sqrt(np.mean(ramp ** 2))
    rms_detrended = np.sqrt(np.mean(detrended ** 2))

    assert rms_detrended < 0.01 * rms_original, (
        f"Linear trend was not removed. "
        f"RMS before={rms_original:.4f}, after={rms_detrended:.6f}"
    )


def test_linear_detrend_output_length_unchanged():
    """Detrended signal must have the same number of samples as input."""
    signal    = make_sine(10.0) + np.linspace(0, 20, N)
    detrended = apply_linear_detrend(signal)
    assert len(detrended) == len(signal)


def test_detrend_dataframe_preserves_metadata():
    """
    Non-EEG columns (time_ms, subject_id, event) must be identical
    before and after DataFrame-level detrending.
    """
    t = np.arange(N) / FS * 1000
    df = pd.DataFrame({
        "time_ms":    t,
        "subject_id": np.ones(N, dtype=int),
        "event":      np.array(["Relaxed"] * N),
        "Fp1":        make_sine(10.0) + np.linspace(0, 20, N),
        "C3":         make_sine(8.0)  + np.linspace(5, -5, N),
    })

    channels     = ["Fp1", "C3"]
    df_detrended = apply_detrend_to_dataframe(df, channels)

    # Metadata must be unchanged
    pd.testing.assert_series_equal(df["time_ms"],    df_detrended["time_ms"])
    pd.testing.assert_series_equal(df["subject_id"], df_detrended["subject_id"])
    pd.testing.assert_series_equal(df["event"],      df_detrended["event"])

    # EEG channels must have been modified
    assert not np.allclose(df["Fp1"].values, df_detrended["Fp1"].values), \
        "Fp1 channel was not detrended."
