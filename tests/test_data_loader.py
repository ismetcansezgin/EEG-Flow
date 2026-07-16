import numpy as np
import pandas as pd
import pytest

from backend.utils.data_loader import EEGDataLoader


def test_data_loader_basic_validation(tmp_path):
    """
    Tests the main responsibilities of EEGDataLoader:

    - Detecting EEG channel columns
    - Excluding metadata columns
    - Estimating the sampling frequency
    - Filling missing values separately for each subject
    """

    test_data = pd.DataFrame(
        {
            "time_ms": [0, 4, 8, 0, 4, 8],
            "subject_id": [1, 1, 1, 2, 2, 2],
            "trial_id": [1, 1, 1, 2, 2, 2],
            "label": [
                "relax",
                "relax",
                "relax",
                "task",
                "task",
                "task",
            ],
            "Fp1": [
                1.0,
                np.nan,
                3.0,
                10.0,
                np.nan,
                14.0,
            ],
            "F3": [
                5.0,
                6.0,
                7.0,
                20.0,
                21.0,
                22.0,
            ],
        }
    )

    csv_path = tmp_path / "sample_eeg.csv"
    test_data.to_csv(csv_path, index=False)

    loader = EEGDataLoader(str(csv_path))
    dataframe, info = loader.load_and_validate()

    # EEG channels should be detected correctly.
    assert info["num_channels"] == 2
    assert info["channels"] == ["Fp1", "F3"]

    # Metadata columns must not be classified as EEG channels.
    assert "time_ms" not in info["channels"]
    assert "subject_id" not in info["channels"]
    assert "trial_id" not in info["channels"]
    assert "label" not in info["channels"]

    # A four-millisecond interval corresponds to 250 Hz.
    assert info["sampling_rate_hz"] == pytest.approx(250.0)

    # Missing values should be interpolated inside each subject.
    assert dataframe.loc[1, "Fp1"] == pytest.approx(2.0)
    assert dataframe.loc[4, "Fp1"] == pytest.approx(12.0)

    # Subject and event columns should be identified.
    assert info["has_subject_id"] is True
    assert info["has_events"] is True