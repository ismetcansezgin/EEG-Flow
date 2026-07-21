"""
backend/utils/filters.py
========================
EEGFlow — Digital Signal Filtering Module

Provides Butterworth Band-pass filter utilities for multi-channel EEG data.

The filter is implemented using Second-Order Sections (SOS) format for
numerical stability and applied with sosfiltfilt() for zero-phase distortion
(i.e., no time-shift in the filtered signal).

Author : EEGFlow Internship Project
Created: Day 6 — 21.07.2026
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, sosfreqz


def design_bandpass_filter(
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4
) -> np.ndarray:
    """
    Designs a Butterworth Band-pass filter in SOS (Second-Order Sections) format.

    A band-pass filter combines:
      - High-pass  (removes slow drifts below `lowcut`  Hz)
      - Low-pass   (removes high-frequency noise above `highcut` Hz)

    SOS format is used instead of (b, a) coefficients because it is numerically
    more stable, especially for higher filter orders.

    Parameters
    ----------
    lowcut  : float
        Lower cut-off frequency in Hz (e.g. 0.5 Hz).
        Frequencies BELOW this value will be removed.
    highcut : float
        Upper cut-off frequency in Hz (e.g. 45.0 Hz).
        Frequencies ABOVE this value will be removed.
    fs      : float
        Sampling frequency of the EEG signal in Hz (e.g. 250 Hz).
    order   : int, optional
        Filter order. Higher = sharper roll-off but more computation. Default: 4.

    Returns
    -------
    sos : np.ndarray
        Second-Order Sections array with shape (n_sections, 6).

    Raises
    ------
    ValueError
        If lowcut >= highcut or if either frequency exceeds the Nyquist limit.
    """
    nyquist = fs / 2.0  # Nyquist frequency: max detectable frequency = fs / 2

    if lowcut <= 0 or highcut <= 0:
        raise ValueError(f"Cut-off frequencies must be positive. Got lowcut={lowcut}, highcut={highcut}")

    if lowcut >= highcut:
        raise ValueError(f"lowcut ({lowcut} Hz) must be less than highcut ({highcut} Hz).")

    if highcut >= nyquist:
        raise ValueError(
            f"highcut ({highcut} Hz) must be less than the Nyquist frequency ({nyquist} Hz). "
            f"Increase fs or reduce highcut."
        )

    # Normalise frequencies to the range [0, 1] where 1 = Nyquist
    low  = lowcut  / nyquist
    high = highcut / nyquist

    # Design the Butterworth band-pass filter in SOS format
    sos = butter(N=order, Wn=[low, high], btype='bandpass', output='sos')
    return sos


def apply_bandpass_filter(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4
) -> np.ndarray:
    """
    Applies a zero-phase Butterworth Band-pass filter to a single EEG channel.

    Uses sosfiltfilt() which runs the filter TWICE (forward + backward),
    resulting in ZERO phase distortion — meaning the filtered signal is
    not shifted in time compared to the original.

    Parameters
    ----------
    signal  : np.ndarray
        1D array of raw EEG amplitude values (in µV) for a single channel.
    lowcut  : float
        Lower cut-off frequency in Hz.
    highcut : float
        Upper cut-off frequency in Hz.
    fs      : float
        Sampling frequency in Hz.
    order   : int, optional
        Filter order. Default: 4.

    Returns
    -------
    filtered : np.ndarray
        1D array of filtered signal values, same length as input.

    Raises
    ------
    ValueError
        If the signal is too short for the filter order.
    """
    # Minimum required signal length: sosfiltfilt needs at least 3 * (2*order) samples
    min_len = 3 * 2 * order
    if len(signal) < min_len:
        raise ValueError(
            f"Signal too short for filtering. "
            f"Minimum length required: {min_len} samples, got: {len(signal)}."
        )

    sos      = design_bandpass_filter(lowcut, highcut, fs, order)
    filtered = sosfiltfilt(sos, signal)
    return filtered


def apply_filter_to_dataframe(
    df: pd.DataFrame,
    channels: list,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4
) -> pd.DataFrame:
    """
    Applies the Butterworth Band-pass filter to every EEG channel column
    in a pandas DataFrame.

    Non-channel columns (e.g. time, subject_id, event) are preserved unchanged.

    Parameters
    ----------
    df       : pd.DataFrame
        Input EEG DataFrame. Must contain columns listed in `channels`.
    channels : list of str
        Column names corresponding to EEG electrode signals to be filtered.
    lowcut   : float
        Lower cut-off frequency in Hz.
    highcut  : float
        Upper cut-off frequency in Hz.
    fs       : float
        Sampling frequency in Hz.
    order    : int, optional
        Filter order. Default: 4.

    Returns
    -------
    df_filtered : pd.DataFrame
        New DataFrame identical to the input but with filtered values in the
        EEG channel columns. Original DataFrame is not modified (copy).
    """
    df_filtered = df.copy()  # Never modify the raw input data

    for channel in channels:
        raw_signal               = df_filtered[channel].values.astype(float)
        df_filtered[channel]     = apply_bandpass_filter(raw_signal, lowcut, highcut, fs, order)

    return df_filtered
