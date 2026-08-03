"""
backend/utils/features.py
=========================
EEGFlow — EEG Signal Feature Extraction Engine (Phase 2)

Transforms 3D epoched EEG matrices (n_epochs, n_channels, n_samples)
into 2D tabular feature DataFrames (n_epochs, n_features) suitable for
machine learning models (Random Forest, SVM, XGBoost, etc.).

Feature Categories Extracted per Channel:
  1. Time-Domain Statistical Features (7 features):
     - Mean
     - Standard Deviation (std)
     - Variance (var)
     - Root Mean Square (rms)
     - Peak-to-Peak Amplitude (ptp)
     - Skewness (skew)
     - Kurtosis (kurtosis)

  2. Frequency-Domain Band Power Features via Welch PSD (12 features):
     - Absolute Band Powers: Delta (0.5-4Hz), Theta (4-8Hz), Alpha (8-13Hz),
       Beta (13-30Hz), Gamma (30-45Hz), Total Power (0.5-45Hz).
     - Relative Band Powers: Ratio of each band power to total power.

Author  : EEGFlow Internship Project
Created : Day 14 — 31.07.2026
"""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats


# Define standard EEG frequency band ranges (Hz)
EEG_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 45.0),
}


def extract_time_domain_features(
    epochs: np.ndarray,
    channel_names: list = None
) -> pd.DataFrame:
    """
    Extracts 7 time-domain statistical features per channel for each epoch window.

    Parameters
    ----------
    epochs : np.ndarray
        Epoched EEG signal array. Accepted shapes:
          - (n_epochs, n_channels, n_samples)
          - (n_channels, n_samples) → treated as 1 epoch
    channel_names : list of str, optional
        Electrode channel names. Defaults to ['CH1', 'CH2', ...].

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (n_epochs, n_channels * 7) containing columns like
        'Fp1_mean', 'Fp1_std', 'Fp1_var', 'Fp1_rms', 'Fp1_ptp', 'Fp1_skew', 'Fp1_kurtosis'.
    """
    # Ensure 3D array: (n_epochs, n_channels, n_samples)
    if epochs.ndim == 2:
        epochs = epochs[np.newaxis, :, :]

    n_epochs, n_channels, n_samples = epochs.shape

    if channel_names is None or len(channel_names) != n_channels:
        channel_names = [f'CH{i+1}' for i in range(n_channels)]

    feature_rows = []

    for ep_idx in range(n_epochs):
        row_dict = {}
        for ch_idx, ch_name in enumerate(channel_names):
            sig = epochs[ep_idx, ch_idx, :]

            sig_std = float(np.std(sig))
            mean_val = float(np.mean(sig))
            std_val  = sig_std
            var_val  = float(np.var(sig))
            rms_val  = float(np.sqrt(np.mean(sig ** 2)))
            ptp_val  = float(np.ptp(sig))

            if sig_std > 1e-9:
                skew_val = float(scipy_stats.skew(sig))
                kurt_val = float(scipy_stats.kurtosis(sig))
            else:
                skew_val = 0.0
                kurt_val = 0.0

            row_dict[f'{ch_name}_mean']     = round(mean_val, 6)
            row_dict[f'{ch_name}_std']      = round(std_val, 6)
            row_dict[f'{ch_name}_var']      = round(var_val, 6)
            row_dict[f'{ch_name}_rms']      = round(rms_val, 6)
            row_dict[f'{ch_name}_ptp']      = round(ptp_val, 6)
            row_dict[f'{ch_name}_skew']     = round(skew_val, 6)
            row_dict[f'{ch_name}_kurtosis'] = round(kurt_val, 6)

        feature_rows.append(row_dict)

    return pd.DataFrame(feature_rows)


def extract_frequency_domain_features(
    epochs: np.ndarray,
    fs: float = 250.0,
    channel_names: list = None
) -> pd.DataFrame:
    """
    Extracts 11 frequency-domain band power features per channel using Welch's PSD.

    Band Powers Extracted:
      - Absolute Powers: Delta, Theta, Alpha, Beta, Gamma, Total Power
      - Relative Powers: Rel_Delta, Rel_Theta, Rel_Alpha, Rel_Beta, Rel_Gamma

    Parameters
    ----------
    epochs : np.ndarray
        Epoched EEG signal array of shape (n_epochs, n_channels, n_samples).
    fs : float
        Sampling frequency in Hz (default: 250.0).
    channel_names : list of str, optional
        Electrode channel names.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (n_epochs, n_channels * 11) containing band power features.
    """
    if epochs.ndim == 2:
        epochs = epochs[np.newaxis, :, :]

    n_epochs, n_channels, n_samples = epochs.shape

    if channel_names is None or len(channel_names) != n_channels:
        channel_names = [f'CH{i+1}' for i in range(n_channels)]

    # Determine segment length for Welch PSD (min of n_samples or 256)
    nperseg = min(n_samples, int(fs * 1.0))

    feature_rows = []

    for ep_idx in range(n_epochs):
        row_dict = {}
        for ch_idx, ch_name in enumerate(channel_names):
            sig = epochs[ep_idx, ch_idx, :]

            # Compute Welch Power Spectral Density
            freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=nperseg)
            freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

            # Compute absolute power for each band
            band_powers = {}
            for band_name, (f_low, f_high) in EEG_BANDS.items():
                band_mask = (freqs >= f_low) & (freqs < f_high if band_name != 'gamma' else freqs <= f_high)
                band_pow  = float(np.sum(psd[band_mask]) * freq_res)
                band_powers[band_name] = band_pow

            total_power = sum(band_powers.values())
            if total_power <= 0:
                total_power = 1e-10

            row_dict[f'{ch_name}_total_power'] = round(total_power, 6)

            for band_name, band_pow in band_powers.items():
                rel_pow = band_pow / total_power
                row_dict[f'{ch_name}_{band_name}_power']     = round(band_pow, 6)
                row_dict[f'{ch_name}_rel_{band_name}_power'] = round(rel_pow, 6)

        feature_rows.append(row_dict)

    return pd.DataFrame(feature_rows)


def extract_all_features(
    epochs: np.ndarray,
    fs: float = 250.0,
    channel_names: list = None,
    labels: list = None,
    subjects: list = None
) -> pd.DataFrame:
    """
    Extracts all 18 time and frequency domain features per channel, combines them
    into a single 2D feature DataFrame, and appends optional event/subject labels.

    Parameters
    ----------
    epochs : np.ndarray
        Epoched EEG signal array of shape (n_epochs, n_channels, n_samples).
    fs : float
        Sampling frequency in Hz (default: 250.0).
    channel_names : list of str, optional
        Electrode channel names.
    labels : list of str, optional
        Majority-vote event class label for each epoch window.
    subjects : list, optional
        Subject IDs for each epoch window.

    Returns
    -------
    pd.DataFrame
        Complete tabular feature matrix of shape (n_epochs, n_channels * 18 + metadata).
    """
    df_time = extract_time_domain_features(epochs, channel_names=channel_names)
    df_freq = extract_frequency_domain_features(epochs, fs=fs, channel_names=channel_names)

    # Combine time and frequency features horizontally
    df_features = pd.concat([df_time, df_freq], axis=1)

    # Append metadata columns if provided
    if subjects is not None and len(subjects) == len(df_features):
        df_features['subject_id'] = subjects

    if labels is not None and len(labels) == len(df_features):
        df_features['event'] = labels

    return df_features
