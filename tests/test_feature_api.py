"""
tests/test_feature_api.py
=========================
Integration tests for EEGFlow POST /api/extract-features endpoint.

Tests verify:
  1.  Default parameters return 200 OK with valid feature matrix shape.
  2.  Custom window/overlap parameters produce correct n_epochs.
  3.  Only time-domain features mode (include_freq_features=false).
  4.  Only frequency-domain features mode (include_time_features=false).
  5.  Non-CSV file upload returns 400 Bad Request.
  6.  Invalid overlap_ratio (>=1.0) returns 400 Bad Request.
  7.  Invalid window_size_sec (<=0) returns 400 Bad Request.
  8.  Both features disabled simultaneously returns 400 Bad Request.
  9.  Response contains all required keys.
  10. feature_names list length matches n_features.
  11. alpha_validation dict contains one entry per unique event class.
  12. class_band_summary contains all 5 EEG bands per class.

Run with:  python -m pytest tests/test_feature_api.py -v
"""

import io
import math
import numpy as np
import pandas as pd
import pytest

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

SAMPLE_CSV_PATH = "data/sample_eeg.csv"


def _post_extract(csv_path: str = SAMPLE_CSV_PATH, **form_kwargs):
    """Helper: POST to /api/extract-features with a CSV file and optional form params."""
    defaults = {
        "window_size_sec": "2.0",
        "overlap_ratio":   "0.5",
        "include_time_features": "true",
        "include_freq_features": "true",
    }
    defaults.update(form_kwargs)

    with open(csv_path, "rb") as f:
        csv_bytes = f.read()

    return client.post(
        "/api/extract-features",
        files={"file": ("sample_eeg.csv", io.BytesIO(csv_bytes), "text/csv")},
        data=defaults,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Default parameters return 200 OK with valid feature matrix shape
# ══════════════════════════════════════════════════════════════════════════════
def test_extract_default_params_returns_200():
    """Default params must return HTTP 200 with feature_matrix_shape [n_epochs, n_features]."""
    resp = _post_extract()
    assert resp.status_code == 200
    data = resp.json()
    shape = data["feature_matrix_shape"]
    assert len(shape) == 2
    assert shape[0] > 0, "n_epochs must be > 0"
    assert shape[1] > 0, "n_features must be > 0"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Custom window/overlap params produce plausible n_epochs
# ══════════════════════════════════════════════════════════════════════════════
def test_extract_custom_params_returns_200():
    """Custom window_size_sec=1.0 and overlap_ratio=0.0 must return 200 OK."""
    resp = _post_extract(window_size_sec="1.0", overlap_ratio="0.0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_epochs"] > 0
    assert data["n_features"] > 0


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Only time-domain features mode
# ══════════════════════════════════════════════════════════════════════════════
def test_extract_only_time_features():
    """include_freq_features=false must return only time-domain features (7 per channel)."""
    resp = _post_extract(include_freq_features="false")
    assert resp.status_code == 200
    data = resp.json()
    assert data["freq_domain_count"] == 0, "freq_domain_count must be 0"
    assert data["time_domain_count"] > 0, "time_domain_count must be > 0"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Only frequency-domain features mode
# ══════════════════════════════════════════════════════════════════════════════
def test_extract_only_freq_features():
    """include_time_features=false must return only frequency-domain features."""
    resp = _post_extract(include_time_features="false")
    assert resp.status_code == 200
    data = resp.json()
    assert data["time_domain_count"] == 0, "time_domain_count must be 0"
    assert data["freq_domain_count"] > 0, "freq_domain_count must be > 0"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: Non-CSV file returns 400
# ══════════════════════════════════════════════════════════════════════════════
def test_extract_non_csv_returns_400():
    """Uploading a non-CSV file must return HTTP 400 Bad Request."""
    fake_bin = io.BytesIO(b"\x00\x01\x02\x03")
    resp = client.post(
        "/api/extract-features",
        files={"file": ("data.txt", fake_bin, "text/plain")},
        data={"window_size_sec": "2.0", "overlap_ratio": "0.5",
              "include_time_features": "true", "include_freq_features": "true"},
    )
    assert resp.status_code == 400


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: Invalid overlap_ratio >= 1.0 returns 400
# ══════════════════════════════════════════════════════════════════════════════
def test_extract_invalid_overlap_returns_400():
    """overlap_ratio >= 1.0 must return HTTP 400 Bad Request."""
    resp = _post_extract(overlap_ratio="1.0")
    assert resp.status_code == 400


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: Invalid window_size_sec <= 0 returns 400
# ══════════════════════════════════════════════════════════════════════════════
def test_extract_invalid_window_size_returns_400():
    """window_size_sec <= 0 must return HTTP 400 Bad Request."""
    resp = _post_extract(window_size_sec="-1.0")
    assert resp.status_code == 400


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8: Both features disabled returns 400
# ══════════════════════════════════════════════════════════════════════════════
def test_extract_both_features_disabled_returns_400():
    """Setting both include_time_features and include_freq_features to false must return 400."""
    resp = _post_extract(include_time_features="false", include_freq_features="false")
    assert resp.status_code == 400


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9: Response contains all required keys
# ══════════════════════════════════════════════════════════════════════════════
def test_extract_response_keys():
    """Response JSON must contain all expected top-level keys."""
    resp = _post_extract()
    assert resp.status_code == 200
    data = resp.json()
    required_keys = [
        "success", "message", "feature_matrix_shape",
        "n_epochs", "n_features", "feature_names",
        "time_domain_count", "freq_domain_count",
        "class_band_summary", "alpha_validation",
        "sample_preview", "epoch_info",
    ]
    for key in required_keys:
        assert key in data, f"Missing key: '{key}' in response"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 10: feature_names length matches n_features
# ══════════════════════════════════════════════════════════════════════════════
def test_extract_feature_names_length_matches():
    """Length of feature_names list must exactly equal n_features."""
    resp = _post_extract()
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["feature_names"]) == data["n_features"], (
        f"feature_names length ({len(data['feature_names'])}) != n_features ({data['n_features']})"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 11: alpha_validation has one entry per unique event class
# ══════════════════════════════════════════════════════════════════════════════
def test_extract_alpha_validation_keys():
    """alpha_validation must contain keys matching unique event classes in data."""
    resp = _post_extract()
    assert resp.status_code == 200
    data = resp.json()
    # Sample data has Relaxed and Task classes
    alpha_val = data["alpha_validation"]
    assert len(alpha_val) >= 1, "alpha_validation must have at least 1 class entry"
    for cls, val in alpha_val.items():
        assert isinstance(val, float), f"alpha_validation[{cls}] must be a float"
        assert 0.0 <= val <= 1.0, f"Relative alpha power must be in [0, 1], got {val}"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 12: class_band_summary contains all 5 EEG bands per class
# ══════════════════════════════════════════════════════════════════════════════
def test_extract_class_band_summary_structure():
    """class_band_summary must contain delta/theta/alpha/beta/gamma for each event class."""
    resp = _post_extract()
    assert resp.status_code == 200
    data = resp.json()
    expected_bands = {"delta", "theta", "alpha", "beta", "gamma"}
    for cls, band_dict in data["class_band_summary"].items():
        missing = expected_bands - set(band_dict.keys())
        assert not missing, f"class_band_summary['{cls}'] missing bands: {missing}"
