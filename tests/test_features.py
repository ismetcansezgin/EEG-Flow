"""
tests/test_features.py
======================
Unit tests for EEGFlow backend/utils/features.py module.

Tests verify:
  1.  Time-domain feature DataFrame shape (n_epochs, n_channels * 7).
  2.  Time-domain accuracy on constant signal (mean=val, std=0, var=0, ptp=0).
  3.  Time-domain accuracy on sine wave (mean ~ 0, ptp = 2 * amp).
  4.  Frequency-domain feature DataFrame shape (n_epochs, n_channels * 11).
  5.  10 Hz pure sine wave concentrates power in Alpha band (rel_alpha > 0.8).
  6.  20 Hz pure sine wave concentrates power in Beta band (rel_beta > 0.8).
  7.  extract_all_features() combines time + freq features correctly.
  8.  Metadata columns (subject_id, event) are appended correctly.
  9.  Single 2D epoch (n_channels, n_samples) returns 1-row DataFrame.
  10. No NaN or Inf values exist in extracted feature matrices.
  11. Custom channel names are reflected in column prefixes.
  12. Relative band powers sum to approximately 1.0 per channel.

Run with:  python -m pytest tests/test_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from backend.utils.features import (
    extract_time_domain_features,
    extract_frequency_domain_features,
    extract_all_features,
)

FS         = 250.0   # Hz
N_EPOCHS   = 10
N_CHANNELS = 8
N_SAMPLES  = 500     # 2.0 s @ 250 Hz
CH_NAMES   = ['Fp1', 'Fp2', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']


@pytest.fixture
def synthetic_epochs():
    """Deterministic 3D synthetic EEG epoch array (10 epochs, 8 channels, 500 samples)."""
    np.random.seed(42)
    return np.random.randn(N_EPOCHS, N_CHANNELS, N_SAMPLES).astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Time-domain features shape
# ══════════════════════════════════════════════════════════════════════════════
def test_extract_time_domain_features_shape(synthetic_epochs):
    """Time-domain feature DataFrame must have shape (n_epochs, n_channels * 7)."""
    df = extract_time_domain_features(synthetic_epochs, channel_names=CH_NAMES)
    expected_cols = N_CHANNELS * 7
    assert df.shape == (N_EPOCHS, expected_cols), (
        f"Expected shape ({N_EPOCHS}, {expected_cols}), got {df.shape}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Constant signal time-domain values
# ══════════════════════════════════════════════════════════════════════════════
def test_time_domain_constant_signal():
    """Constant signal x(t)=5.0 must yield mean=5, std=0, var=0, rms=5, ptp=0."""
    const_epoch = np.full((1, 1, 500), 5.0)
    df = extract_time_domain_features(const_epoch, channel_names=['CH1'])
    assert df.loc[0, 'CH1_mean'] == 5.0
    assert df.loc[0, 'CH1_std']  == 0.0
    assert df.loc[0, 'CH1_var']  == 0.0
    assert df.loc[0, 'CH1_rms']  == 5.0
    assert df.loc[0, 'CH1_ptp']  == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Sine wave time-domain values
# ══════════════════════════════════════════════════════════════════════════════
def test_time_domain_sine_wave():
    """Sine wave A=3.0 must yield mean ~ 0 and ptp ~ 6.0."""
    t = np.arange(500) / FS
    sine_sig = 3.0 * np.sin(2 * np.pi * 10 * t)
    epoch = sine_sig.reshape(1, 1, 500)
    df = extract_time_domain_features(epoch, channel_names=['Fp1'])
    assert abs(df.loc[0, 'Fp1_mean']) < 0.05
    assert abs(df.loc[0, 'Fp1_ptp'] - 6.0) < 0.1


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Frequency-domain features shape
# ══════════════════════════════════════════════════════════════════════════════
def test_extract_frequency_domain_features_shape(synthetic_epochs):
    """Frequency-domain feature DataFrame must have shape (n_epochs, n_channels * 11)."""
    df = extract_frequency_domain_features(synthetic_epochs, fs=FS, channel_names=CH_NAMES)
    expected_cols = N_CHANNELS * 11
    assert df.shape == (N_EPOCHS, expected_cols), (
        f"Expected shape ({N_EPOCHS}, {expected_cols}), got {df.shape}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: 10 Hz sine wave concentrates power in Alpha band
# ══════════════════════════════════════════════════════════════════════════════
def test_frequency_domain_alpha_sine_wave():
    """Pure 10 Hz sine wave (Alpha band: 8-13 Hz) must have rel_alpha_power > 0.8."""
    t = np.arange(500) / FS
    alpha_sig = 5.0 * np.sin(2 * np.pi * 10 * t)
    epoch = alpha_sig.reshape(1, 1, 500)
    df = extract_frequency_domain_features(epoch, fs=FS, channel_names=['Fp1'])
    rel_alpha = df.loc[0, 'Fp1_rel_alpha_power']
    assert rel_alpha > 0.8, f"Expected rel_alpha > 0.8 for 10 Hz sine, got {rel_alpha}"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: 20 Hz sine wave concentrates power in Beta band
# ══════════════════════════════════════════════════════════════════════════════
def test_frequency_domain_beta_sine_wave():
    """Pure 20 Hz sine wave (Beta band: 13-30 Hz) must have rel_beta_power > 0.8."""
    t = np.arange(500) / FS
    beta_sig = 5.0 * np.sin(2 * np.pi * 20 * t)
    epoch = beta_sig.reshape(1, 1, 500)
    df = extract_frequency_domain_features(epoch, fs=FS, channel_names=['C3'])
    rel_beta = df.loc[0, 'C3_rel_beta_power']
    assert rel_beta > 0.8, f"Expected rel_beta > 0.8 for 20 Hz sine, got {rel_beta}"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: extract_all_features combines time + freq features
# ══════════════════════════════════════════════════════════════════════════════
def test_extract_all_features_combined_columns(synthetic_epochs):
    """extract_all_features must combine 7 time + 11 freq = 18 features per channel."""
    df = extract_all_features(synthetic_epochs, fs=FS, channel_names=CH_NAMES)
    expected_cols = N_CHANNELS * 18
    assert df.shape == (N_EPOCHS, expected_cols), (
        f"Expected {expected_cols} columns, got {df.shape[1]}"
    )
    assert 'Fp1_mean' in df.columns
    assert 'Fp1_alpha_power' in df.columns
    assert 'O2_rel_gamma_power' in df.columns


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8: Metadata columns appended correctly
# ══════════════════════════════════════════════════════════════════════════════
def test_extract_all_features_with_metadata(synthetic_epochs):
    """Providing labels and subjects must append 'event' and 'subject_id' columns."""
    labels   = ['Relaxed'] * 5 + ['Active'] * 5
    subjects = [1] * 5 + [2] * 5
    df = extract_all_features(synthetic_epochs, fs=FS, channel_names=CH_NAMES, labels=labels, subjects=subjects)
    assert 'event' in df.columns
    assert 'subject_id' in df.columns
    assert list(df['event']) == labels
    assert list(df['subject_id']) == subjects


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9: Single 2D epoch input
# ══════════════════════════════════════════════════════════════════════════════
def test_single_epoch_2d_input():
    """2D input (n_channels, n_samples) must return 1-row DataFrame."""
    epoch_2d = np.random.randn(8, 500)
    df = extract_all_features(epoch_2d, fs=FS, channel_names=CH_NAMES)
    assert df.shape[0] == 1, f"Expected 1 row for 2D input, got {df.shape[0]}"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 10: No NaN or Inf values in extracted features
# ══════════════════════════════════════════════════════════════════════════════
def test_no_nan_or_inf_in_features(synthetic_epochs):
    """Extracted feature DataFrame must not contain any NaN or Inf values."""
    df = extract_all_features(synthetic_epochs, fs=FS, channel_names=CH_NAMES)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    assert not df[numeric_cols].isna().any().any(), "Feature matrix contains NaN values."
    assert not np.isinf(df[numeric_cols].values).any(), "Feature matrix contains Inf values."


# ══════════════════════════════════════════════════════════════════════════════
# TEST 11: Custom channel names preserved
# ══════════════════════════════════════════════════════════════════════════════
def test_custom_channel_names_preserved(synthetic_epochs):
    """Column names must use provided channel prefixes."""
    custom_ch = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'FC5', 'FC6']
    df = extract_all_features(synthetic_epochs, fs=FS, channel_names=custom_ch)
    assert 'AF3_mean' in df.columns
    assert 'FC6_rel_alpha_power' in df.columns


# ══════════════════════════════════════════════════════════════════════════════
# TEST 12: Relative band powers sum to ~1.0
# ══════════════════════════════════════════════════════════════════════════════
def test_relative_powers_sum_to_one(synthetic_epochs):
    """Sum of relative band powers (delta+theta+alpha+beta+gamma) must equal ~1.0."""
    df = extract_frequency_domain_features(synthetic_epochs, fs=FS, channel_names=CH_NAMES)
    for ch in CH_NAMES:
        rel_sum = (
            df[f'{ch}_rel_delta_power'] +
            df[f'{ch}_rel_theta_power'] +
            df[f'{ch}_rel_alpha_power'] +
            df[f'{ch}_rel_beta_power']  +
            df[f'{ch}_rel_gamma_power']
        )
        np.testing.assert_allclose(rel_sum.values, 1.0, atol=0.05, err_msg=f"Relative power sum for {ch} is not 1.0")
