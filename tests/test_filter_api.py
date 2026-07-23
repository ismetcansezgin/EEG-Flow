"""
tests/test_filter_api.py
========================
Integration tests for the EEGFlow POST /api/filter endpoint.

Tests verify:
  1.  Valid CSV with all filters enabled returns 200 and correct structure.
  2.  Valid CSV with only bandpass filter active.
  3.  Valid CSV with only notch filter active.
  4.  Valid CSV with only detrend filter active.
  5.  Valid CSV with all filters disabled returns raw == filtered stats.
  6.  Non-CSV file upload returns 400 error.
  7.  Response contains correct top-level keys.
  8.  Filtered statistics differ from raw statistics when filters are applied.
  9.  pipeline_config reflects the filter parameters sent in the request.
  10. All 8 EEG channels are present in both raw and filtered statistics.

Run with:  python -m pytest tests/test_filter_api.py -v
"""

import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# Path to the shared sample dataset created on Day 4
SAMPLE_CSV = "data/sample_eeg.csv"
EXPECTED_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2"]


def _post_filter(extra_data: dict = None):
    """Helper: POST to /api/filter with sample_eeg.csv and optional form overrides."""
    data = {
        "apply_bandpass":  "true",
        "lowcut":          "0.5",
        "highcut":         "45.0",
        "order":           "4",
        "apply_notch":     "true",
        "notch_freq":      "50.0",
        "quality_factor":  "30.0",
        "apply_detrend":   "true",
    }
    if extra_data:
        data.update(extra_data)

    with open(SAMPLE_CSV, "rb") as f:
        return client.post(
            "/api/filter",
            files={"file": ("sample_eeg.csv", f, "text/csv")},
            data=data,
        )


# ══════════════════════════════════════════════════════════
# TEST 1: All filters enabled — 200 + correct structure
# ══════════════════════════════════════════════════════════
def test_filter_all_enabled_returns_200():
    """POST /api/filter with all filters enabled must return HTTP 200."""
    assert os.path.exists(SAMPLE_CSV), "sample_eeg.csv must exist before running test."

    response = _post_filter()
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "pipeline_config"      in data
    assert "raw_statistics"       in data
    assert "filtered_statistics"  in data
    assert "sample_preview"       in data


# ══════════════════════════════════════════════════════════
# TEST 2: Only bandpass filter active
# ══════════════════════════════════════════════════════════
def test_filter_only_bandpass():
    """Endpoint must return 200 when only bandpass is enabled."""
    response = _post_filter({
        "apply_bandpass": "true",
        "apply_notch":    "false",
        "apply_detrend":  "false",
    })
    assert response.status_code == 200
    config = response.json()["pipeline_config"]["applied_filters"]
    assert config["bandpass"]["active"] is True
    assert config["notch"]["active"]    is False
    assert config["detrend"]            is False


# ══════════════════════════════════════════════════════════
# TEST 3: Only notch filter active
# ══════════════════════════════════════════════════════════
def test_filter_only_notch():
    """Endpoint must return 200 when only notch filter is enabled."""
    response = _post_filter({
        "apply_bandpass": "false",
        "apply_notch":    "true",
        "apply_detrend":  "false",
    })
    assert response.status_code == 200
    config = response.json()["pipeline_config"]["applied_filters"]
    assert config["bandpass"]["active"] is False
    assert config["notch"]["active"]    is True
    assert config["detrend"]            is False


# ══════════════════════════════════════════════════════════
# TEST 4: Only detrend active
# ══════════════════════════════════════════════════════════
def test_filter_only_detrend():
    """Endpoint must return 200 when only detrend is enabled."""
    response = _post_filter({
        "apply_bandpass": "false",
        "apply_notch":    "false",
        "apply_detrend":  "true",
    })
    assert response.status_code == 200
    config = response.json()["pipeline_config"]["applied_filters"]
    assert config["bandpass"]["active"] is False
    assert config["notch"]["active"]    is False
    assert config["detrend"]            is True


# ══════════════════════════════════════════════════════════
# TEST 5: All filters disabled — raw stats ≈ filtered stats
# ══════════════════════════════════════════════════════════
def test_filter_all_disabled_stats_unchanged():
    """When all filters are disabled, filtered stats must equal raw stats."""
    response = _post_filter({
        "apply_bandpass": "false",
        "apply_notch":    "false",
        "apply_detrend":  "false",
    })
    assert response.status_code == 200
    data = response.json()

    for ch in EXPECTED_CHANNELS:
        raw_rms      = data["raw_statistics"][ch]["rms"]
        filtered_rms = data["filtered_statistics"][ch]["rms"]
        assert raw_rms == filtered_rms, (
            f"Channel {ch}: expected raw RMS == filtered RMS when no filters applied, "
            f"got raw={raw_rms}, filtered={filtered_rms}"
        )


# ══════════════════════════════════════════════════════════
# TEST 6: Non-CSV file returns 400
# ══════════════════════════════════════════════════════════
def test_filter_non_csv_returns_400():
    """Uploading a non-CSV file must return HTTP 400."""
    fake_content = b"not a csv file"
    response = client.post(
        "/api/filter",
        files={"file": ("signal.txt", fake_content, "text/plain")},
        data={"apply_bandpass": "true", "apply_notch": "false", "apply_detrend": "false"},
    )
    assert response.status_code == 400


# ══════════════════════════════════════════════════════════
# TEST 7: All 8 EEG channels present in response
# ══════════════════════════════════════════════════════════
def test_filter_all_channels_present_in_response():
    """All 8 EEG channels must appear in raw and filtered statistics."""
    response = _post_filter()
    assert response.status_code == 200
    data = response.json()

    for ch in EXPECTED_CHANNELS:
        assert ch in data["raw_statistics"],      f"Channel {ch} missing from raw_statistics."
        assert ch in data["filtered_statistics"], f"Channel {ch} missing from filtered_statistics."
        assert ch in data["sample_preview"],      f"Channel {ch} missing from sample_preview."


# ══════════════════════════════════════════════════════════
# TEST 8: Filtering changes signal statistics
# ══════════════════════════════════════════════════════════
def test_filter_changes_signal_statistics():
    """
    With all filters enabled the filtered RMS must differ from the raw RMS
    for at least one channel, confirming the pipeline actually modifies the signal.
    """
    response = _post_filter()
    assert response.status_code == 200
    data = response.json()

    any_changed = any(
        data["raw_statistics"][ch]["rms"] != data["filtered_statistics"][ch]["rms"]
        for ch in EXPECTED_CHANNELS
    )
    assert any_changed, "No channel statistics changed after filtering — pipeline may not be working."


# ══════════════════════════════════════════════════════════
# TEST 9: pipeline_config reflects sent parameters
# ══════════════════════════════════════════════════════════
def test_filter_pipeline_config_reflects_params():
    """Response pipeline_config must match the filter parameters sent in the request."""
    response = _post_filter({
        "lowcut":      "1.0",
        "highcut":     "40.0",
        "notch_freq":  "60.0",
    })
    assert response.status_code == 200
    config = response.json()["pipeline_config"]["applied_filters"]

    assert config["bandpass"]["lowcut"]     == 1.0
    assert config["bandpass"]["highcut"]    == 40.0
    assert config["notch"]["notch_freq"]    == 60.0


# ══════════════════════════════════════════════════════════
# TEST 10: sample_preview contains 100 samples per channel
# ══════════════════════════════════════════════════════════
def test_filter_sample_preview_length():
    """Each channel in sample_preview must contain exactly 100 sample values."""
    response = _post_filter()
    assert response.status_code == 200
    preview = response.json()["sample_preview"]

    for ch in EXPECTED_CHANNELS:
        assert len(preview[ch]) == 100, (
            f"Channel {ch} preview expected 100 samples, got {len(preview[ch])}."
        )
