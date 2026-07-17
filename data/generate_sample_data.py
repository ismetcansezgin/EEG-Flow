from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

SAMPLING_RATE = 250
DURATION_PER_EVENT_SECONDS = 10

SUBJECT_IDS = [1, 2, 3]
EVENTS = ["Relaxed", "Task"]

EEG_CHANNELS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "O1",
    "O2",
]

RANDOM_SEED = 42
MISSING_VALUE_RATIO = 0.001

OUTPUT_FILE = Path(__file__).parent / "sample_eeg.csv"


# Representative frequencies for common EEG bands.
BAND_FREQUENCIES: Dict[str, float] = {
    "delta": 2.0,
    "theta": 6.0,
    "alpha": 10.0,
    "beta": 20.0,
    "gamma": 35.0,
}


# Base amplitudes used for the simulated EEG bands.
BASE_BAND_AMPLITUDES: Dict[str, float] = {
    "delta": 4.0,
    "theta": 5.0,
    "alpha": 8.0,
    "beta": 3.0,
    "gamma": 1.5,
}


# Approximate channel-dependent scaling factors.
# Occipital channels receive stronger alpha activity.
CHANNEL_PROFILES: Dict[str, Dict[str, float]] = {
    "Fp1": {
        "delta": 1.20,
        "theta": 1.10,
        "alpha": 0.70,
        "beta": 1.10,
        "gamma": 1.00,
    },
    "Fp2": {
        "delta": 1.20,
        "theta": 1.10,
        "alpha": 0.70,
        "beta": 1.10,
        "gamma": 1.00,
    },
    "F3": {
        "delta": 1.00,
        "theta": 1.10,
        "alpha": 0.85,
        "beta": 1.20,
        "gamma": 1.00,
    },
    "F4": {
        "delta": 1.00,
        "theta": 1.10,
        "alpha": 0.85,
        "beta": 1.20,
        "gamma": 1.00,
    },
    "C3": {
        "delta": 0.90,
        "theta": 1.00,
        "alpha": 1.00,
        "beta": 1.10,
        "gamma": 1.00,
    },
    "C4": {
        "delta": 0.90,
        "theta": 1.00,
        "alpha": 1.00,
        "beta": 1.10,
        "gamma": 1.00,
    },
    "O1": {
        "delta": 0.80,
        "theta": 0.90,
        "alpha": 1.60,
        "beta": 0.80,
        "gamma": 0.80,
    },
    "O2": {
        "delta": 0.80,
        "theta": 0.90,
        "alpha": 1.60,
        "beta": 0.80,
        "gamma": 0.80,
    },
}


def generate_band_signal(
    time_seconds: np.ndarray,
    frequency: float,
    amplitude: float,
    phase: float,
) -> np.ndarray:
    """
    Generates a sinusoidal signal for one EEG frequency band.

    Args:
        time_seconds:
            Time points in seconds.
        frequency:
            Frequency of the signal in Hz.
        amplitude:
            Signal amplitude.
        phase:
            Phase offset in radians.

    Returns:
        Generated sinusoidal signal.
    """

    return amplitude * np.sin(
        2.0 * np.pi * frequency * time_seconds + phase
    )


def get_event_band_amplitudes(event: str) -> Dict[str, float]:
    """
    Returns EEG band amplitudes according to the simulated event.

    Relaxed represents an eyes-closed condition with stronger alpha
    activity. Task represents an eyes-open condition with alpha
    blocking and slightly stronger beta activity.

    Args:
        event:
            Event label, either Relaxed or Task.

    Returns:
        Dictionary containing event-specific band amplitudes.
    """

    amplitudes = BASE_BAND_AMPLITUDES.copy()

    if event == "Relaxed":
        # Eyes closed: strong alpha activity.
        amplitudes["alpha"] = 15.0
        amplitudes["beta"] = 2.5

    elif event == "Task":
        # Eyes open/task: alpha suppression and increased beta.
        amplitudes["alpha"] = 4.0
        amplitudes["beta"] = 5.0
        amplitudes["gamma"] = 2.0

    else:
        raise ValueError(f"Unsupported event type: {event}")

    return amplitudes


def generate_eeg_channel(
    time_seconds: np.ndarray,
    channel: str,
    event: str,
    subject_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generates one synthetic EEG channel.

    The output combines Delta, Theta, Alpha, Beta and Gamma
    oscillations with Gaussian noise, slow baseline drift and
    50 Hz line noise.

    Args:
        time_seconds:
            Time points in seconds.
        channel:
            EEG channel name.
        event:
            Event label.
        subject_scale:
            Small scaling value used to differentiate subjects.
        rng:
            NumPy random number generator.

    Returns:
        Simulated EEG signal as a NumPy array.
    """

    if channel not in CHANNEL_PROFILES:
        raise ValueError(f"Unknown EEG channel: {channel}")

    event_amplitudes = get_event_band_amplitudes(event)
    channel_profile = CHANNEL_PROFILES[channel]

    signal = np.zeros_like(time_seconds, dtype=float)

    for band_name, frequency in BAND_FREQUENCIES.items():
        phase = rng.uniform(0.0, 2.0 * np.pi)

        amplitude_variation = rng.uniform(0.90, 1.10)

        amplitude = (
            event_amplitudes[band_name]
            * channel_profile[band_name]
            * subject_scale
            * amplitude_variation
        )

        band_signal = generate_band_signal(
            time_seconds=time_seconds,
            frequency=frequency,
            amplitude=amplitude,
            phase=phase,
        )

        signal += band_signal

    # Add a slow baseline drift.
    drift_frequency = rng.uniform(0.10, 0.30)
    drift_amplitude = rng.uniform(0.5, 1.5)
    drift_phase = rng.uniform(0.0, 2.0 * np.pi)

    baseline_drift = generate_band_signal(
        time_seconds=time_seconds,
        frequency=drift_frequency,
        amplitude=drift_amplitude,
        phase=drift_phase,
    )

    # Add 50 Hz power-line interference.
    line_noise_phase = rng.uniform(0.0, 2.0 * np.pi)

    line_noise = generate_band_signal(
        time_seconds=time_seconds,
        frequency=50.0,
        amplitude=rng.uniform(0.3, 0.8),
        phase=line_noise_phase,
    )

    # Add Gaussian measurement noise.
    gaussian_noise = rng.normal(
        loc=0.0,
        scale=2.0,
        size=len(time_seconds),
    )

    signal = (
        signal
        + baseline_drift
        + line_noise
        + gaussian_noise
    )

    return signal


def generate_subject_event_data(
    subject_id: int,
    event: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generates all EEG channels for one subject and one event.

    Args:
        subject_id:
            Participant identifier.
        event:
            Relaxed or Task event.
        rng:
            NumPy random number generator.

    Returns:
        DataFrame containing time, subject, event and EEG channels.
    """

    number_of_samples = (
        SAMPLING_RATE * DURATION_PER_EVENT_SECONDS
    )

    time_seconds = np.arange(
        number_of_samples,
        dtype=float,
    ) / SAMPLING_RATE

    time_ms = time_seconds * 1000.0

    # A small difference is introduced between subjects.
    subject_scale = rng.uniform(0.90, 1.10)

    data = {
        "time_ms": time_ms,
        "subject_id": np.full(
            number_of_samples,
            subject_id,
            dtype=int,
        ),
        "event": np.full(
            number_of_samples,
            event,
            dtype=object,
        ),
    }

    for channel in EEG_CHANNELS:
        data[channel] = generate_eeg_channel(
            time_seconds=time_seconds,
            channel=channel,
            event=event,
            subject_scale=subject_scale,
            rng=rng,
        )

    return pd.DataFrame(data)


def inject_missing_values(
    dataframe: pd.DataFrame,
    rng: np.random.Generator,
    missing_ratio: float = MISSING_VALUE_RATIO,
) -> int:
    """
    Inserts sparse NaN values into EEG channel columns.

    Missing values are not inserted at the first or last sample of
    each subject-event segment. This allows linear interpolation to
    operate correctly.

    Args:
        dataframe:
            Complete synthetic EEG dataset.
        rng:
            NumPy random number generator.
        missing_ratio:
            Ratio of EEG values that will be replaced with NaN.

    Returns:
        Total number of inserted missing values.
    """

    if not 0.0 <= missing_ratio < 1.0:
        raise ValueError(
            "missing_ratio must be between 0 and 1."
        )

    inserted_missing_values = 0

    grouped_indices = dataframe.groupby(
        ["subject_id", "event"],
        sort=False,
    ).indices

    for channel in EEG_CHANNELS:
        for group_indices in grouped_indices.values():
            group_indices = np.asarray(group_indices)

            # Do not select the first and last samples.
            valid_indices = group_indices[1:-1]

            number_of_missing_values = max(
                1,
                int(len(valid_indices) * missing_ratio),
            )

            selected_indices = rng.choice(
                valid_indices,
                size=number_of_missing_values,
                replace=False,
            )

            dataframe.loc[selected_indices, channel] = np.nan

            inserted_missing_values += len(selected_indices)

    return inserted_missing_values


def generate_dataset(
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generates the complete multi-subject EEG dataset.

    Args:
        rng:
            NumPy random number generator.

    Returns:
        Combined EEG dataset.
    """

    generated_segments: List[pd.DataFrame] = []

    for subject_id in SUBJECT_IDS:
        for event in EVENTS:
            segment = generate_subject_event_data(
                subject_id=subject_id,
                event=event,
                rng=rng,
            )

            generated_segments.append(segment)

    dataset = pd.concat(
        generated_segments,
        ignore_index=True,
    )

    return dataset


def print_dataset_summary(
    dataframe: pd.DataFrame,
    missing_value_count: int,
) -> None:
    """
    Prints basic information about the generated dataset.
    """

    print("\nSynthetic EEG dataset generated successfully.")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Number of rows: {len(dataframe)}")
    print(f"Number of subjects: {dataframe['subject_id'].nunique()}")
    print(f"Number of EEG channels: {len(EEG_CHANNELS)}")
    print(f"Sampling rate: {SAMPLING_RATE} Hz")
    print(
        "Duration per event: "
        f"{DURATION_PER_EVENT_SECONDS} seconds"
    )
    print(f"Events: {', '.join(EVENTS)}")
    print(f"Inserted missing values: {missing_value_count}")

    print("\nMissing values per EEG channel:")

    missing_per_channel = (
        dataframe[EEG_CHANNELS]
        .isna()
        .sum()
    )

    print(missing_per_channel.to_string())

    print("\nRows per subject and event:")

    group_counts = (
        dataframe.groupby(
            ["subject_id", "event"],
            sort=False,
        )
        .size()
        .rename("number_of_rows")
    )

    print(group_counts.to_string())


def main() -> None:
    """
    Generates and saves the synthetic EEG dataset.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    dataset = generate_dataset(rng)

    missing_value_count = inject_missing_values(
        dataframe=dataset,
        rng=rng,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
        float_format="%.6f",
    )

    print_dataset_summary(
        dataframe=dataset,
        missing_value_count=missing_value_count,
    )


if __name__ == "__main__":
    main()