"""
tests/test_epoch_api.py
======================
Integration tests for the EEGFlow POST /api/epoch REST API endpoint.

Tests verify:
  1.  Valid CSV with default parameters (2.0 s, 50% overlap) returns 200 OK and expected structure.
  2.  Custom parameters (1.0 s window, 25% overlap) return 200 OK and updated shape.
  3.  Non-overlapping mode (0.0 overlap) returns 200 OK and correct epoch count.
  4.  High overlap mode (75% overlap) returns 200 OK and double epoch count.
  5.  Uploading a non-CSV file returns 400 Bad Request error.
  6.  Invalid overlap_ratio (>= 1.0) returns 400 Bad Request error.
  7.  Invalid window_size_sec (<= 0) returns 400 Bad Request error.
  8.  Response JSON contains all required top-level keys.
  9.  Length of returned labels array matches n_epochs exactly.

Run with:  python -m pytest tests/test_epoch_api.py -v
"""

import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

SAMPLE_CSV = "data/sample_eeg.csv"


def _post_epoch(extra_data: dict = None):
    """Helper: POST to /api/epoch with sample_eeg.csv and optional form overrides."""
    data = {
        "window_size_sec": "2.0",
        "overlap_ratio":   "0.5",
    }
    if extra_data:
        data.update(extra_data)

    with open(SAMPLE_CSV, "rb") as f:
        return client.post(
            "/api/epoch",
            files={"file": ("sample_eeg.csv", f, "text/csv")},
            data=data,
        )


# ══════════════════════════════════════════════════════════
# TEST 1: Default parameters → 200 OK & correct structure
# ══════════════════════════════════════════════════════════
def test_epoch_default_params_returns_200():
    """POST /api/epoch with default params (2.0s, 0.5) must return HTTP 200."""
    assert os.path.exists(SAMPLE_CSV), "sample_eeg.csv must exist before running test."

    response = _post_epoch()
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "n_epochs"             in data
    assert "epoch_shape"          in data
    assert "labels"               in data
    assert "epoch_info"           in data
    assert "sample_epoch_preview" in data

    # 15000 samples @ 250 Hz = 60s total
    # window = 500 samples (2.0s), step = 250 samples (50% overlap)
    # (15000 - 500) // 250 + 1 = 59 epochs
    assert data["n_epochs"] == 59
    assert data["epoch_shape"] == [59, 8, 500]


# ══════════════════════════════════════════════════════════
# TEST 2: Custom window size and overlap
# ══════════════════════════════════════════════════════════
def test_epoch_custom_params_returns_200():
    """Custom window_size_sec=1.0 and overlap_ratio=0.25 must return 200 OK."""
    response = _post_epoch({
        "window_size_sec": "1.0",
        "overlap_ratio":   "0.25",
    })
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    # window = 250 samples (1.0s), step = 187 samples (75% of 250)
    # (15000 - 250) // 187 + 1 = 79 epochs
    assert data["epoch_info"]["window_size_sec"] == 1.0
    assert data["epoch_info"]["overlap_ratio"]   == 0.25
    assert data["epoch_shape"][2]                == 250


# ══════════════════════════════════════════════════════════
# TEST 3: Zero overlap (non-overlapping windows)
# ══════════════════════════════════════════════════════════
def test_epoch_zero_overlap_returns_200():
    """overlap_ratio=0.0 (non-overlapping) must return 200 OK and floor(N/W) epochs."""
    response = _post_epoch({
        "window_size_sec": "2.0",
        "overlap_ratio":   "0.0",
    })
    assert response.status_code == 200

    data = response.json()
    # 15000 / 500 = 30 non-overlapping epochs
    assert data["n_epochs"] == 30
    assert data["epoch_shape"] == [30, 8, 500]


# ══════════════════════════════════════════════════════════
# TEST 4: High overlap (75% overlap)
# ══════════════════════════════════════════════════════════
def test_epoch_high_overlap_returns_200():
    """overlap_ratio=0.75 must return 200 OK and increased epoch count."""
    response = _post_epoch({
        "window_size_sec": "2.0",
        "overlap_ratio":   "0.75",
    })
    assert response.status_code == 200

    data = response.json()
    # step = 125 samples → (15000 - 500) // 125 + 1 = 117 epochs
    assert data["n_epochs"] == 117
    assert data["epoch_shape"] == [117, 8, 500]


# ══════════════════════════════════════════════════════════
# TEST 5: Non-CSV file returns 400
# ══════════════════════════════════════════════════════════
def test_epoch_non_csv_returns_400():
    """Uploading a non-CSV file must return HTTP 400 Bad Request."""
    fake_content = b"not a csv file"
    response = client.post(
        "/api/epoch",
        files={"file": ("data.txt", fake_content, "text/plain")},
        data={"window_size_sec": "2.0", "overlap_ratio": "0.5"},
    )
    assert response.status_code == 400
    assert "Only CSV files" in response.json()["detail"]


# ══════════════════════════════════════════════════════════
# TEST 6: Invalid overlap_ratio (>= 1.0) returns 400
# ══════════════════════════════════════════════════════════
def test_epoch_invalid_overlap_ratio_returns_400():
    """overlap_ratio >= 1.0 must return HTTP 400 Bad Request."""
    response = _post_epoch({"overlap_ratio": "1.5"})
    assert response.status_code == 400
    assert "overlap_ratio" in response.json()["detail"]


# ══════════════════════════════════════════════════════════
# TEST 7: Invalid window_size_sec (<= 0) returns 400
# ══════════════════════════════════════════════════════════
def test_epoch_invalid_window_size_returns_400():
    """window_size_sec <= 0 must return HTTP 400 Bad Request."""
    response = _post_epoch({"window_size_sec": "-1.0"})
    assert response.status_code == 400
    assert "window_size_sec" in response.json()["detail"]


# ══════════════════════════════════════════════════════════
# TEST 8: Response JSON contains all required top-level keys
# ══════════════════════════════════════════════════════════
def test_epoch_response_keys():
    """Response must contain all required metadata and preview keys."""
    response = _post_epoch()
    assert response.status_code == 200
    data = response.json()

    required_keys = {
        "success", "message", "n_epochs", "epoch_shape",
        "labels", "subjects", "epoch_info", "sample_epoch_preview"
    }
    missing = required_keys - set(data.keys())
    assert not missing, f"Response missing required keys: {missing}"


# ══════════════════════════════════════════════════════════
# TEST 9: Length of labels array matches n_epochs
# ══════════════════════════════════════════════════════════
def test_epoch_labels_length_matches_n_epochs():
    """Returned labels array length must exactly match n_epochs."""
    response = _post_epoch()
    assert response.status_code == 200
    data = response.json()

    assert len(data["labels"]) == data["n_epochs"], (
        f"Expected {data['n_epochs']} labels, got {len(data['labels'])}"
    )
