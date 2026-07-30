import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
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
from backend.utils.epoching import create_epochs_from_dataframe

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


@app.post("/api/epoch")
async def epoch_eeg_file(
    file: UploadFile = File(...),
    window_size_sec: str = Form("2.0"),
    overlap_ratio: str = Form("0.5"),
):
    """
    POST /api/epoch
    ===============
    Accepts an uploaded raw/filtered EEG CSV file and sliding window parameters,
    executes time-window segmentation (epoching), and returns 3D epoch dimensions,
    majority-vote event/subject labels, and epoch metadata.

    Form Parameters:
        - file: CSV file containing multi-channel EEG data.
        - window_size_sec: Window duration in seconds (default: 2.0).
        - overlap_ratio: Window overlap fraction between 0.0 and 0.9 (default: 0.5).

    Returns:
        JSON response with epoch_shape, n_epochs, labels, subjects, and epoch_info.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.filename}'. Only CSV files (.csv) are supported."
        )

    # Validate parameters
    try:
        w_sec = float(window_size_sec)
        o_ratio = float(overlap_ratio)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Parameters 'window_size_sec' and 'overlap_ratio' must be valid numeric values."
        )

    if w_sec <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid window_size_sec={w_sec}. Must be greater than 0."
        )

    if not 0.0 <= o_ratio < 1.0:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid overlap_ratio={o_ratio}. Must be in range [0.0, 1.0)."
        )

    temp_file_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        # Load and validate CSV data
        loader = EEGDataLoader(temp_file_path)
        df_raw, info = loader.load_and_validate()

        channels = info.get("channels", [])
        fs = float(info.get("sampling_rate_hz", 250.0))

        if not channels:
            raise HTTPException(
                status_code=400,
                detail="No numeric EEG channel columns detected in CSV."
            )

        # Execute epoch segmentation
        epoch_res = create_epochs_from_dataframe(
            df=df_raw,
            channels=channels,
            fs=fs,
            window_size_sec=w_sec,
            overlap_ratio=o_ratio,
        )

        epochs_matrix = epoch_res["epochs"]   # shape: (n_epochs, n_channels, samples_per_epoch)

        # Prepare first 2 epochs preview for frontend visualization
        sample_preview = {}
        n_preview = min(2, epoch_res["n_epochs"])
        for ep_idx in range(n_preview):
            ep_dict = {}
            for ch_idx, ch_name in enumerate(channels):
                # First 50 samples of this epoch for this channel
                ep_dict[ch_name] = [
                    round(float(v), 4) for v in epochs_matrix[ep_idx, ch_idx, :50]
                ]
            sample_preview[f"epoch_{ep_idx + 1}"] = ep_dict

        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        # Convert numpy int64/types in subjects list to native Python int/None for JSON serialization
        raw_subjects = epoch_res.get("subjects")
        clean_subjects = None
        if raw_subjects is not None:
            clean_subjects = [
                int(s) if (s is not None and not pd.isna(s)) else None
                for s in raw_subjects
            ]

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "EEG signal epoching successfully executed.",
                "n_epochs": int(epoch_res["n_epochs"]),
                "epoch_shape": [int(dim) for dim in epochs_matrix.shape],
                "labels": [str(l) for l in epoch_res["labels"]],
                "subjects": clean_subjects,
                "epoch_info": {
                    "window_size_sec": float(epoch_res["epoch_info"]["window_size_sec"]),
                    "overlap_ratio": float(epoch_res["epoch_info"]["overlap_ratio"]),
                    "fs": float(epoch_res["epoch_info"]["fs"]),
                    "n_channels": int(epoch_res["epoch_info"]["n_channels"]),
                    "samples_per_epoch": int(epoch_res["epoch_info"]["samples_per_epoch"]),
                },
                "sample_epoch_preview": sample_preview,
            }
        )

    except HTTPException:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise
    except ValueError as ve:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute signal epoching: {str(e)}"
        )
