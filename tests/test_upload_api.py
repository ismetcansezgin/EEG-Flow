import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_check_endpoint():
    """
    Tests the GET /api/health endpoint to ensure server reports online status.
    """
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["service"] == "EEGFlow API"

def test_upload_eeg_file_endpoint():
    """
    Tests the POST /api/upload endpoint using the generated sample_eeg.csv dataset.
    """
    sample_csv_path = "data/sample_eeg.csv"
    assert os.path.exists(sample_csv_path), "sample_eeg.csv must exist before running test"

    with open(sample_csv_path, "rb") as f:
        response = client.post(
            "/api/upload",
            files={"file": ("sample_eeg.csv", f, "text/csv")}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "metadata" in data
    
    metadata = data["metadata"]
    assert metadata["num_samples"] == 15000
    assert metadata["num_channels"] == 8
    assert metadata["sampling_rate_hz"] == 250.0
    assert metadata["has_subject_id"] is True
