"""
tests/test_epoching.py
======================
Unit tests for EEGFlow backend/utils/epoching.py module.

Tests verify:
  1.  1D signal epochs have the correct shape.
  2.  2D multi-channel signal epochs have the correct shape.
  3.  Epoch count is mathematically correct (N_epochs formula).
  4.  Zero overlap (non-overlapping windows) produces correct epoch count.
  5.  75% overlap produces correct epoch count.
  6.  Epoch data content is identical to the original signal slice.
  7.  Signal shorter than one window raises ValueError.
  8.  Invalid overlap_ratio (>= 1.0) raises ValueError.
  9.  create_epochs_from_dataframe() returns correct shape.
  10. Event labels are correctly aligned to each epoch.
  11. Subject IDs are correctly aligned to each epoch.
  12. DataFrame without event/subject columns returns 'unknown' and None.
  13. epoch_info dict contains all required keys.

Run with:  python -m pytest tests/test_epoching.py -v
"""

import numpy as np
import pandas as pd
import pytest

from backend.utils.epoching import create_epochs, create_epochs_from_dataframe


# ─── Shared Fixtures ──────────────────────────────────────────────────────────

FS             = 250          # Hz
WIN_SEC        = 2.0          # seconds → 500 samples per epoch
WIN_SAMPLES    = int(WIN_SEC * FS)   # 500
OVERLAP        = 0.5          # 50% → step = 250 samples
N_SAMPLES      = 5000         # total samples  (20 s of signal)
N_CHANNELS     = 8

# Expected epoch count for default params:
# step = 500 * (1 - 0.5) = 250
# N_epochs = (5000 - 500) // 250 + 1 = 18 + 1 = 19
EXPECTED_N_EPOCHS = (N_SAMPLES - WIN_SAMPLES) // int(WIN_SAMPLES * (1 - OVERLAP)) + 1


@pytest.fixture
def signal_1d():
    """Deterministic 1D EEG-like signal."""
    np.random.seed(42)
    return np.random.randn(N_SAMPLES).astype(np.float64)


@pytest.fixture
def signal_2d():
    """Deterministic 2D multi-channel signal (n_channels × n_samples)."""
    np.random.seed(42)
    return np.random.randn(N_CHANNELS, N_SAMPLES).astype(np.float64)


@pytest.fixture
def eeg_dataframe(signal_2d):
    """EEG DataFrame with 8 channels, subject_id, and event columns."""
    ch_names = [f'CH{i+1}' for i in range(N_CHANNELS)]
    df = pd.DataFrame(signal_2d.T, columns=ch_names)
    df.insert(0, 'time_ms', np.arange(N_SAMPLES) * (1000 / FS))
    df.insert(1, 'subject_id', [1] * (N_SAMPLES // 2) + [2] * (N_SAMPLES // 2))
    df.insert(2, 'event', ['Relaxed'] * (N_SAMPLES // 2) + ['Active'] * (N_SAMPLES // 2))
    return df, ch_names


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: 1D signal → correct epoch shape
# ══════════════════════════════════════════════════════════════════════════════
def test_create_epochs_1d_shape(signal_1d):
    """1D input must return (n_epochs, n_samples_per_epoch) shaped array."""
    epochs = create_epochs(signal_1d, fs=FS, window_size_sec=WIN_SEC, overlap_ratio=OVERLAP)
    assert epochs.ndim == 2
    assert epochs.shape == (EXPECTED_N_EPOCHS, WIN_SAMPLES), (
        f"Expected shape ({EXPECTED_N_EPOCHS}, {WIN_SAMPLES}), got {epochs.shape}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: 2D signal → correct epoch shape
# ══════════════════════════════════════════════════════════════════════════════
def test_create_epochs_2d_shape(signal_2d):
    """2D input must return (n_epochs, n_channels, n_samples_per_epoch) shaped array."""
    epochs = create_epochs(signal_2d, fs=FS, window_size_sec=WIN_SEC, overlap_ratio=OVERLAP)
    assert epochs.ndim == 3
    assert epochs.shape == (EXPECTED_N_EPOCHS, N_CHANNELS, WIN_SAMPLES), (
        f"Expected shape ({EXPECTED_N_EPOCHS}, {N_CHANNELS}, {WIN_SAMPLES}), got {epochs.shape}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Epoch count matches N_epochs formula
# ══════════════════════════════════════════════════════════════════════════════
def test_epoch_count_formula(signal_2d):
    """Epoch count must satisfy: N_epochs = (N_total - N_win) // N_step + 1."""
    epochs = create_epochs(signal_2d, fs=FS, window_size_sec=WIN_SEC, overlap_ratio=OVERLAP)
    step = int(WIN_SAMPLES * (1 - OVERLAP))
    expected = (N_SAMPLES - WIN_SAMPLES) // step + 1
    assert epochs.shape[0] == expected, (
        f"Expected {expected} epochs, got {epochs.shape[0]}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Zero overlap (non-overlapping windows) epoch count
# ══════════════════════════════════════════════════════════════════════════════
def test_epoch_count_no_overlap(signal_2d):
    """With overlap_ratio=0.0, epochs must be non-overlapping floor(N/W) windows."""
    epochs = create_epochs(signal_2d, fs=FS, window_size_sec=WIN_SEC, overlap_ratio=0.0)
    # step = win_samples = 500, N_epochs = (5000 - 500) // 500 + 1 = 10
    expected = (N_SAMPLES - WIN_SAMPLES) // WIN_SAMPLES + 1
    assert epochs.shape[0] == expected, (
        f"Expected {expected} epochs for 0% overlap, got {epochs.shape[0]}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: 75% overlap epoch count
# ══════════════════════════════════════════════════════════════════════════════
def test_epoch_count_75_overlap(signal_2d):
    """With overlap_ratio=0.75, step = 125 samples, verify epoch count formula."""
    overlap = 0.75
    step = int(WIN_SAMPLES * (1 - overlap))          # 125
    expected = (N_SAMPLES - WIN_SAMPLES) // step + 1  # (4500) // 125 + 1 = 37
    epochs = create_epochs(signal_2d, fs=FS, window_size_sec=WIN_SEC, overlap_ratio=overlap)
    assert epochs.shape[0] == expected, (
        f"Expected {expected} epochs for 75% overlap, got {epochs.shape[0]}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: Epoch data values match original signal slice
# ══════════════════════════════════════════════════════════════════════════════
def test_epoch_data_matches_original_signal(signal_2d):
    """First epoch must contain exactly the first window_samples of the signal."""
    epochs = create_epochs(signal_2d, fs=FS, window_size_sec=WIN_SEC, overlap_ratio=OVERLAP)
    np.testing.assert_array_almost_equal(
        epochs[0],
        signal_2d[:, :WIN_SAMPLES],
        decimal=10,
        err_msg="First epoch data does not match original signal slice.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: Signal shorter than one window raises ValueError
# ══════════════════════════════════════════════════════════════════════════════
def test_signal_shorter_than_window_raises_error():
    """Signal shorter than one epoch window must raise ValueError."""
    short_signal = np.random.randn(100)   # 100 samples < 500 (2.0 s × 250 Hz)
    with pytest.raises(ValueError, match="shorter than one epoch"):
        create_epochs(short_signal, fs=FS, window_size_sec=WIN_SEC, overlap_ratio=OVERLAP)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8: Invalid overlap_ratio raises ValueError
# ══════════════════════════════════════════════════════════════════════════════
def test_invalid_overlap_ratio_raises_error(signal_1d):
    """overlap_ratio >= 1.0 must raise ValueError."""
    with pytest.raises(ValueError, match="overlap_ratio"):
        create_epochs(signal_1d, fs=FS, window_size_sec=WIN_SEC, overlap_ratio=1.0)

    with pytest.raises(ValueError, match="overlap_ratio"):
        create_epochs(signal_1d, fs=FS, window_size_sec=WIN_SEC, overlap_ratio=1.5)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9: DataFrame epochs have correct shape
# ══════════════════════════════════════════════════════════════════════════════
def test_dataframe_epochs_shape(eeg_dataframe):
    """create_epochs_from_dataframe() must return correct (n_epochs, n_ch, n_samples) shape."""
    df, ch_names = eeg_dataframe
    result = create_epochs_from_dataframe(df, ch_names, fs=FS, window_size_sec=WIN_SEC, overlap_ratio=OVERLAP)
    assert result['epochs'].shape == (EXPECTED_N_EPOCHS, N_CHANNELS, WIN_SAMPLES), (
        f"Expected shape ({EXPECTED_N_EPOCHS}, {N_CHANNELS}, {WIN_SAMPLES}), "
        f"got {result['epochs'].shape}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 10: Event labels are correctly aligned to epochs
# ══════════════════════════════════════════════════════════════════════════════
def test_event_labels_alignment(eeg_dataframe):
    """First half epochs must be 'Relaxed', second half must be 'Active'."""
    df, ch_names = eeg_dataframe
    result = create_epochs_from_dataframe(df, ch_names, fs=FS, window_size_sec=WIN_SEC, overlap_ratio=OVERLAP)
    labels = result['labels']
    assert len(labels) == result['n_epochs'], "Labels list length must equal n_epochs."
    assert labels[0] == 'Relaxed', f"First epoch label expected 'Relaxed', got '{labels[0]}'."
    assert labels[-1] == 'Active', f"Last epoch label expected 'Active', got '{labels[-1]}'."


# ══════════════════════════════════════════════════════════════════════════════
# TEST 11: Subject IDs are correctly aligned to epochs
# ══════════════════════════════════════════════════════════════════════════════
def test_subject_ids_alignment(eeg_dataframe):
    """First epoch must belong to subject 1, last epoch to subject 2."""
    df, ch_names = eeg_dataframe
    result = create_epochs_from_dataframe(df, ch_names, fs=FS, window_size_sec=WIN_SEC, overlap_ratio=OVERLAP)
    subjects = result['subjects']
    assert subjects is not None, "subjects should not be None when subject_id column exists."
    assert subjects[0] == 1,  f"First epoch expected subject 1, got {subjects[0]}."
    assert subjects[-1] == 2, f"Last epoch expected subject 2, got {subjects[-1]}."


# ══════════════════════════════════════════════════════════════════════════════
# TEST 12: DataFrame without event/subject columns returns defaults
# ══════════════════════════════════════════════════════════════════════════════
def test_dataframe_no_metadata_columns():
    """Without event/subject_id columns, labels must be 'unknown' and subjects None."""
    np.random.seed(42)
    df = pd.DataFrame({
        'Fp1': np.random.randn(N_SAMPLES),
        'C3' : np.random.randn(N_SAMPLES),
    })
    result = create_epochs_from_dataframe(df, ['Fp1', 'C3'], fs=FS, window_size_sec=WIN_SEC, overlap_ratio=OVERLAP)
    assert all(lbl == 'unknown' for lbl in result['labels']), (
        "All labels must be 'unknown' when no event column is present."
    )
    assert result['subjects'] is None, "subjects must be None when no subject_id column is present."


# ══════════════════════════════════════════════════════════════════════════════
# TEST 13: epoch_info dict contains all required keys
# ══════════════════════════════════════════════════════════════════════════════
def test_epoch_info_keys(eeg_dataframe):
    """epoch_info must contain all required metadata keys."""
    df, ch_names = eeg_dataframe
    result = create_epochs_from_dataframe(df, ch_names, fs=FS, window_size_sec=WIN_SEC, overlap_ratio=OVERLAP)
    required_keys = {'window_size_sec', 'overlap_ratio', 'fs', 'n_channels', 'samples_per_epoch'}
    missing = required_keys - set(result['epoch_info'].keys())
    assert not missing, f"epoch_info is missing required keys: {missing}"
    assert result['epoch_info']['fs']              == FS
    assert result['epoch_info']['n_channels']      == N_CHANNELS
    assert result['epoch_info']['window_size_sec'] == WIN_SEC
    assert result['epoch_info']['overlap_ratio']   == OVERLAP
    assert result['epoch_info']['samples_per_epoch'] == WIN_SAMPLES
