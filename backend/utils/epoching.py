"""
backend/utils/epoching.py
=========================
EEGFlow — EEG Epoch (Sliding Window) Segmentation Module

Divides continuous multi-channel EEG signals into fixed-duration
time windows (epochs) suitable for feature extraction and ML classification.

The sliding window approach allows:
  - Controlled temporal resolution via window_size_sec
  - Data augmentation via overlap_ratio (50% overlap doubles epoch count)
  - Label alignment: each epoch inherits its majority-class event label

Mathematical formula for epoch count:
    N_step   = window_size_samples * (1 - overlap_ratio)
    N_epochs = floor((N_total - N_window) / N_step) + 1

Author  : EEGFlow Internship Project
Created : Day 11 — 28.07.2026
"""

import numpy as np
import pandas as pd


def create_epochs(
    signal: np.ndarray,
    fs: float,
    window_size_sec: float = 2.0,
    overlap_ratio: float = 0.5,
) -> np.ndarray:
    """
    Divides a 1D or 2D (channels × samples) EEG signal array into
    fixed-duration, overlapping epochs using a sliding window.

    Parameters
    ----------
    signal : np.ndarray
        EEG data to epoch. Accepted shapes:
          - (n_samples,)                 → single-channel 1D signal
          - (n_channels, n_samples)      → multi-channel 2D signal
    fs : float
        Sampling frequency in Hz (e.g. 250 Hz).
    window_size_sec : float
        Duration of each epoch window in seconds (e.g. 2.0 s).
        Default: 2.0 s
    overlap_ratio : float
        Fraction of window overlap between consecutive epochs (0.0–0.9).
        0.0 = no overlap (non-overlapping windows).
        0.5 = 50% overlap (step = half the window size).
        Default: 0.5

    Returns
    -------
    epochs : np.ndarray
        Epoched signal array of shape:
          - (n_epochs, n_samples_per_epoch)          if input is 1D
          - (n_epochs, n_channels, n_samples_per_epoch) if input is 2D

    Raises
    ------
    ValueError
        If overlap_ratio is outside [0.0, 0.9].
        If window_size_sec results in 0 samples.
        If signal is shorter than one epoch window.

    Examples
    --------
    >>> import numpy as np
    >>> sig = np.random.randn(8, 5000)   # 8 channels, 5000 samples at 250 Hz
    >>> epochs = create_epochs(sig, fs=250, window_size_sec=2.0, overlap_ratio=0.5)
    >>> epochs.shape
    (7, 8, 500)
    """
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError(
            f"overlap_ratio must be in [0.0, 1.0). Got {overlap_ratio}."
        )

    window_samples = int(window_size_sec * fs)
    if window_samples <= 0:
        raise ValueError(
            f"window_size_sec={window_size_sec} with fs={fs} yields "
            f"{window_samples} samples. Must be > 0."
        )

    # Handle 1D vs 2D input uniformly
    is_1d = signal.ndim == 1
    if is_1d:
        data = signal[np.newaxis, :]   # (1, n_samples)
    else:
        data = signal                  # (n_channels, n_samples)

    n_channels, n_total = data.shape

    if n_total < window_samples:
        raise ValueError(
            f"Signal length ({n_total} samples) is shorter than one epoch "
            f"window ({window_samples} samples = {window_size_sec} s × {fs} Hz)."
        )

    step_samples = max(1, int(window_samples * (1.0 - overlap_ratio)))

    # Calculate number of complete epochs
    n_epochs = (n_total - window_samples) // step_samples + 1

    # Pre-allocate output array
    epochs = np.zeros((n_epochs, n_channels, window_samples), dtype=data.dtype)

    for i in range(n_epochs):
        start = i * step_samples
        end   = start + window_samples
        epochs[i] = data[:, start:end]

    # Squeeze back to 2D if input was 1D
    if is_1d:
        epochs = epochs[:, 0, :]   # (n_epochs, n_samples_per_epoch)

    return epochs


def create_epochs_from_dataframe(
    df: pd.DataFrame,
    channels: list,
    fs: float,
    window_size_sec: float = 2.0,
    overlap_ratio: float = 0.5,
) -> dict:
    """
    Segments a multi-channel EEG DataFrame into epochs using a sliding window
    and aligns each epoch with its majority-class event and subject labels.

    Parameters
    ----------
    df : pd.DataFrame
        Input EEG DataFrame. Must contain the columns listed in `channels`.
        Optional metadata columns: 'subject_id', 'event'.
    channels : list of str
        EEG electrode column names to epoch (e.g. ['Fp1', 'C3', 'O1']).
    fs : float
        Sampling frequency in Hz (e.g. 250 Hz).
    window_size_sec : float
        Duration of each epoch window in seconds. Default: 2.0 s.
    overlap_ratio : float
        Fraction of window overlap between consecutive epochs (0.0–0.9).
        Default: 0.5

    Returns
    -------
    result : dict with the following keys:
        'epochs'     : np.ndarray, shape (n_epochs, n_channels, n_samples_per_epoch)
        'labels'     : list of str, majority-class event label for each epoch.
                       'unknown' if no 'event' column found.
        'subjects'   : list, subject_id for each epoch (majority vote).
                       None if no 'subject_id' column found.
        'n_epochs'   : int, total number of epochs produced.
        'epoch_info' : dict with window_size_sec, overlap_ratio, fs, n_channels.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> t   = np.arange(5000) / 250
    >>> df  = pd.DataFrame({'time_ms': t*1000, 'subject_id': [1]*5000,
    ...                     'event': ['Relaxed']*5000,
    ...                     'Fp1': np.sin(2*np.pi*10*t),
    ...                     'C3' : np.sin(2*np.pi*8*t)})
    >>> res = create_epochs_from_dataframe(df, ['Fp1', 'C3'], fs=250)
    >>> res['epochs'].shape
    (7, 2, 500)
    """
    # Extract channel data as (n_channels, n_samples)
    signal_matrix = df[channels].values.T.astype(float)

    epochs = create_epochs(
        signal=signal_matrix,
        fs=fs,
        window_size_sec=window_size_sec,
        overlap_ratio=overlap_ratio,
    )

    n_epochs = epochs.shape[0]
    window_samples = int(window_size_sec * fs)
    step_samples   = max(1, int(window_samples * (1.0 - overlap_ratio)))

    # ── Align event labels via majority vote per epoch window ──
    labels = []
    subjects = []

    has_events  = 'event'      in df.columns
    has_subject = 'subject_id' in df.columns

    for i in range(n_epochs):
        start = i * step_samples
        end   = start + window_samples

        if has_events:
            window_events = df['event'].iloc[start:end]
            label = window_events.mode()[0] if not window_events.empty else 'unknown'
            labels.append(str(label))
        else:
            labels.append('unknown')

        if has_subject:
            window_subjects = df['subject_id'].iloc[start:end]
            subject = window_subjects.mode()[0] if not window_subjects.empty else None
            subjects.append(subject)
        else:
            subjects.append(None)

    return {
        'epochs':     epochs,
        'labels':     labels,
        'subjects':   subjects if has_subject else None,
        'n_epochs':   n_epochs,
        'epoch_info': {
            'window_size_sec': window_size_sec,
            'overlap_ratio':   overlap_ratio,
            'fs':              fs,
            'n_channels':      len(channels),
            'samples_per_epoch': window_samples,
        },
    }
