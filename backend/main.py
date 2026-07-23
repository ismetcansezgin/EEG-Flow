import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.utils.data_loader import EEGDataLoader
from backend.utils.filters import (
    apply_filter_to_dataframe,
    apply_notch_to_dataframe,
    apply_detrend_to_dataframe,
)

# Resolve the frontend directory relative to this file
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI(
    title="EEGFlow API",
    description="Backend REST API for EEG Signal Processing and Machine Learning Pipeline",
    version="1.0.0"
)

# Enable CORS to allow requests from any frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend/index.html at the root URL
@app.get("/", include_in_schema=False)
async def serve_frontend():
    """
    Serves the main dashboard HTML file at the root URL.
    Access the dashboard at: http://127.0.0.1:8000
    """
    index_path = FRONTEND_DIR / "index.html"
    return FileResponse(str(index_path))

# Mount all other static frontend assets (CSS, JS)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint to verify backend API status.
    """
    return {
        "status": "online",
        "service": "EEGFlow API",
        "version": "1.0.0"
    }

@app.post("/api/upload")
async def upload_eeg_file(file: UploadFile = File(...)):
    """
    Endpoint to receive uploaded EEG CSV files, execute data loader validation,
    perform NaN interpolation, and return structured metadata info.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only CSV files (.csv) are supported."
        )

    temp_file_path = None
    try:
        # Save uploaded file to a temporary system directory
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"upload_{file.filename}")
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the file using EEGDataLoader
        loader = EEGDataLoader(temp_file_path)
        _, info = loader.load_and_validate()

        # Clean up temporary file after processing
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "EEG CSV file successfully uploaded and validated.",
                "metadata": info
            }
        )

    except Exception as e:
        # Clean up temporary file in case of error
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process EEG file: {str(e)}"
        )


@app.post("/api/filter")
async def filter_eeg_file(
    file: UploadFile = File(...),
    apply_bandpass: str = Form(default="true"),
    lowcut: str = Form(default="0.5"),
    highcut: str = Form(default="45.0"),
    order: str = Form(default="4"),
    apply_notch: str = Form(default="true"),
    notch_freq: str = Form(default="50.0"),
    quality_factor: str = Form(default="30.0"),
    apply_detrend: str = Form(default="true"),
):
    """
    Endpoint to receive an EEG CSV file, execute requested signal filters
    (Band-pass, Notch, Linear Detrending), and return filtered channel
    statistics and signals.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only CSV files (.csv) are supported."
        )

    temp_file_path = None
    try:
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"filter_{file.filename}")

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Load and validate CSV data
        loader = EEGDataLoader(temp_file_path)
        df_raw, info = loader.load_and_validate()

        channels = info.get("channels", [])
        fs = float(info.get("sampling_rate_hz", 250.0))

        if not channels:
            raise HTTPException(
                status_code=400,
                detail="No numeric EEG channel columns detected in CSV."
            )

        # Parse filter configuration parameters
        do_bp = apply_bandpass.lower() == "true"
        do_notch = apply_notch.lower() == "true"
        do_detrend = apply_detrend.lower() == "true"

        l_cut = float(lowcut)
        h_cut = float(highcut)
        filt_order = int(order)
        n_freq = float(notch_freq)
        q_fact = float(quality_factor)

        # 2. Record raw channel statistics (before filtering)
        raw_stats = {}
        for ch in channels:
            signal_raw = df_raw[ch].values.astype(float)
            raw_stats[ch] = {
                "mean": round(float(np.mean(signal_raw)), 4),
                "std": round(float(np.std(signal_raw)), 4),
                "rms": round(float(np.sqrt(np.mean(signal_raw ** 2))), 4),
                "peak_to_peak": round(float(np.ptp(signal_raw)), 4),
            }

        # 3. Apply digital filters sequentially
        df_filtered = df_raw.copy()

        if do_detrend:
            df_filtered = apply_detrend_to_dataframe(df_filtered, channels)

        if do_bp:
            df_filtered = apply_filter_to_dataframe(
                df_filtered, channels, lowcut=l_cut, highcut=h_cut, fs=fs, order=filt_order
            )

        if do_notch:
            df_filtered = apply_notch_to_dataframe(
                df_filtered, channels, notch_freq=n_freq, fs=fs, quality_factor=q_fact
            )

        # 4. Record filtered channel statistics (after filtering)
        filtered_stats = {}
        filtered_channels_data = {}
        for ch in channels:
            signal_filt = df_filtered[ch].values.astype(float)
            filtered_stats[ch] = {
                "mean": round(float(np.mean(signal_filt)), 4),
                "std": round(float(np.std(signal_filt)), 4),
                "rms": round(float(np.sqrt(np.mean(signal_filt ** 2))), 4),
                "peak_to_peak": round(float(np.ptp(signal_filt)), 4),
            }
            # Provide sample preview (first 100 samples)
            filtered_channels_data[ch] = [round(float(v), 4) for v in signal_filt[:100]]

        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "EEG signal filtering successfully executed.",
                "pipeline_config": {
                    "sampling_rate_hz": fs,
                    "applied_filters": {
                        "detrend": do_detrend,
                        "bandpass": {
                            "active": do_bp,
                            "lowcut": l_cut,
                            "highcut": h_cut,
                            "order": filt_order
                        },
                        "notch": {
                            "active": do_notch,
                            "notch_freq": n_freq,
                            "quality_factor": q_fact
                        }
                    }
                },
                "raw_statistics": raw_stats,
                "filtered_statistics": filtered_stats,
                "sample_preview": filtered_channels_data
            }
        )

    except HTTPException:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise
    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute signal filtering: {str(e)}"
        )
