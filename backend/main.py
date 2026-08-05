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
from backend.utils.features import extract_all_features
from backend.utils.models import CLASSIFIERS, train_evaluate, cross_validate_model

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


@app.post("/api/extract-features")
async def extract_eeg_features(
    file: UploadFile = File(...),
    window_size_sec: str = Form("2.0"),
    overlap_ratio: str = Form("0.5"),
    include_time_features: str = Form("true"),
    include_freq_features: str = Form("true"),
):
    """
    POST /api/extract-features
    ==========================
    Accepts an uploaded EEG CSV file, segments it into epochs using a sliding
    window, extracts time-domain and/or frequency-domain features per epoch,
    and returns the feature matrix metadata with per-class band power summaries
    for Alpha wave validation visualization (Milestone 2).

    Form Parameters:
        - file: CSV file containing multi-channel EEG data.
        - window_size_sec: Window duration in seconds (default: 2.0).
        - overlap_ratio: Window overlap fraction 0.0-0.9 (default: 0.5).
        - include_time_features: Include time-domain features (default: true).
        - include_freq_features: Include frequency-domain PSD features (default: true).

    Returns:
        JSON with feature_matrix_shape, feature_names, per-class band power
        summaries, and sample feature previews.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.filename}'. Only CSV files (.csv) are supported."
        )

    try:
        w_sec    = float(window_size_sec)
        o_ratio  = float(overlap_ratio)
        do_time  = include_time_features.lower() == "true"
        do_freq  = include_freq_features.lower() == "true"
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Parameters must be valid numeric or boolean values."
        )

    if w_sec <= 0:
        raise HTTPException(status_code=400, detail=f"Invalid window_size_sec={w_sec}. Must be > 0.")
    if not 0.0 <= o_ratio < 1.0:
        raise HTTPException(status_code=400, detail=f"Invalid overlap_ratio={o_ratio}. Must be in [0.0, 1.0).")
    if not do_time and not do_freq:
        raise HTTPException(status_code=400, detail="At least one of include_time_features or include_freq_features must be true.")

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file_path = tmp.name

        loader = EEGDataLoader(temp_file_path)
        df_raw, info = loader.load_and_validate()

        channels = info.get("channels", [])
        fs = float(info.get("sampling_rate_hz", 250.0))

        if not channels:
            raise HTTPException(status_code=400, detail="No numeric EEG channel columns detected in CSV.")

        # Step 1: Epoch the signal
        epoch_res = create_epochs_from_dataframe(
            df=df_raw, channels=channels, fs=fs,
            window_size_sec=w_sec, overlap_ratio=o_ratio,
        )
        epochs_matrix = epoch_res["epochs"]
        labels   = epoch_res["labels"]
        subjects = epoch_res["subjects"]

        # Step 2: Extract features
        df_features = extract_all_features(
            epochs   = epochs_matrix,
            fs       = fs,
            channel_names = channels,
            labels   = labels if labels else None,
            subjects = [int(s) if (s is not None and not pd.isna(s)) else None for s in subjects] if subjects else None,
        )

        # Filter feature columns by user selection
        time_suffixes = ["_mean", "_std", "_var", "_rms", "_ptp", "_skew", "_kurtosis"]
        freq_suffixes = ["_power", "_total_power"]
        metadata_cols = ["event", "subject_id"]

        selected_cols = []
        for col in df_features.columns:
            if col in metadata_cols:
                selected_cols.append(col)
                continue
            is_time = any(col.endswith(s) for s in time_suffixes)
            is_freq = any(s in col for s in freq_suffixes)
            if (do_time and is_time) or (do_freq and is_freq):
                selected_cols.append(col)

        df_selected = df_features[selected_cols]
        feature_cols = [c for c in selected_cols if c not in metadata_cols]
        n_features = len(feature_cols)
        n_epochs   = len(df_selected)

        # Step 3: Compute per-class band power summaries for Alpha-wave validation
        eeg_bands = ["delta", "theta", "alpha", "beta", "gamma"]
        class_band_summary = {}
        unique_labels = list(set(labels)) if labels else []

        for lbl in unique_labels:
            lbl_mask = [i for i, l in enumerate(labels) if l == lbl]
            lbl_df   = df_features.iloc[lbl_mask]
            band_avgs = {}
            for band in eeg_bands:
                band_cols = [c for c in feature_cols if f"_{band}_power" in c and "rel_" not in c]
                if band_cols:
                    mean_power = float(lbl_df[band_cols].values.mean())
                    band_avgs[band] = round(mean_power, 6)
            class_band_summary[str(lbl)] = band_avgs

        # Step 4: Relative alpha per class (for Alpha wave validation chart)
        alpha_validation = {}
        for lbl in unique_labels:
            lbl_mask = [i for i, l in enumerate(labels) if l == lbl]
            lbl_df   = df_features.iloc[lbl_mask]
            rel_alpha_cols = [c for c in feature_cols if "rel_alpha_power" in c]
            if rel_alpha_cols:
                mean_rel_alpha = float(lbl_df[rel_alpha_cols].values.mean())
                alpha_validation[str(lbl)] = round(mean_rel_alpha, 6)

        # Step 5: Sample feature preview (first 2 epochs, first 5 features)
        preview_features = feature_cols[:5]
        sample_preview = []
        for i in range(min(2, n_epochs)):
            row = {col: round(float(df_selected.iloc[i][col]), 4) for col in preview_features}
            if "event" in df_selected.columns:
                row["event"] = str(df_selected.iloc[i]["event"])
            sample_preview.append(row)

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "EEG feature extraction successfully executed.",
                "feature_matrix_shape": [n_epochs, n_features],
                "n_epochs":   n_epochs,
                "n_features": n_features,
                "feature_names": feature_cols,
                "time_domain_count": len([c for c in feature_cols if any(c.endswith(s) for s in time_suffixes)]),
                "freq_domain_count": len([c for c in feature_cols if any(s in c for s in freq_suffixes)]),
                "class_band_summary":  class_band_summary,
                "alpha_validation":    alpha_validation,
                "sample_preview":      sample_preview,
                "epoch_info": {
                    "window_size_sec":  float(w_sec),
                    "overlap_ratio":    float(o_ratio),
                    "fs":               float(fs),
                    "n_channels":       int(len(channels)),
                    "samples_per_epoch": int(round(w_sec * fs)),
                },
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
            detail=f"Failed to execute feature extraction: {str(e)}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: POST /api/train-model
# Phase 3 — Train SVM / Random Forest / XGBoost on extracted EEG feature matrix
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/train-model")
async def train_model_endpoint(
    file: UploadFile = File(..., description="EEG dataset CSV file (with event_label column)"),
    model_name: str  = Form("svm",  description="Classifier: 'svm' | 'random_forest' | 'xgboost'"),
    window_size_sec: float = Form(2.0, description="Epoch window size in seconds"),
    overlap_ratio:   float = Form(0.5, description="Sliding window overlap ratio [0, 1)"),
    test_size:       float = Form(0.2, description="Test split fraction [0.1, 0.5]"),
    random_state:    int   = Form(42,  description="Random seed for reproducibility"),
):
    """
    Full ML pipeline in one API call:
      CSV upload → epoch segmentation → feature extraction → model training → evaluation metrics.

    Returns:
    --------
    JSON with accuracy, precision, recall, F1, confusion matrix, classification report,
    train/test sample counts, and per-model metadata.
    """
    # ── Parameter validation ──────────────────────────────────────────────────
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    if model_name not in CLASSIFIERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'. Choose from: {CLASSIFIERS}"
        )

    if not (0.1 <= test_size <= 0.5):
        raise HTTPException(status_code=400, detail="test_size must be between 0.1 and 0.5.")

    if window_size_sec <= 0:
        raise HTTPException(status_code=400, detail="window_size_sec must be greater than 0.")

    if not (0.0 <= overlap_ratio < 1.0):
        raise HTTPException(status_code=400, detail="overlap_ratio must be in [0.0, 1.0).")

    temp_file_path = None
    try:
        # ── Save uploaded CSV to temp file ─────────────────────────────────
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_file_path = tmp.name

        # ── Load & validate EEG data ────────────────────────────────────────
        loader = EEGDataLoader(temp_file_path)
        df, metadata = loader.load_and_validate()

        sampling_rate = metadata.get("sampling_rate_hz", metadata.get("sampling_rate", 250))
        eeg_channels  = metadata.get("channels", metadata.get("eeg_channels", []))

        if not eeg_channels:
            raise HTTPException(status_code=400, detail="No EEG channels found in the uploaded CSV.")

        # ── Epoch segmentation ──────────────────────────────────────────────
        epoch_result = create_epochs_from_dataframe(
            df=df,
            channels=eeg_channels,
            fs=sampling_rate,
            window_size_sec=window_size_sec,
            overlap_ratio=overlap_ratio,
        )

        epochs_array = epoch_result["epochs"]    # shape: (n_epochs, n_channels, samples)
        labels_list  = epoch_result.get("labels", None)
        n_epochs     = epoch_result["n_epochs"]

        if n_epochs == 0:
            raise HTTPException(status_code=400, detail="No epochs could be created. Check window_size_sec and data length.")

        feature_df = extract_all_features(
            epochs=epochs_array,
            fs=sampling_rate,
            channel_names=eeg_channels,
            labels=labels_list,
        )
        feature_names = [c for c in feature_df.columns if c not in ("event", "event_label", "subject_id", "subject")]

        # ── Prepare X (features) and y (labels) ─────────────────────────────
        # extract_all_features stores labels in 'event' column
        label_col = "event" if "event" in feature_df.columns else "event_label"
        if label_col not in feature_df.columns:
            if labels_list is not None:
                feature_df["event"] = labels_list
                label_col = "event"
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No event label column found. The CSV must contain an event_label column."
                )

        X = feature_df[feature_names].values.astype(np.float32)
        y = feature_df[label_col].values

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"At least 2 event classes required. Found: {unique_classes.tolist()}"
            )

        # ── Train & evaluate ─────────────────────────────────────────────────
        metrics = train_evaluate(
            X=X,
            y=y,
            model_name=model_name,
            test_size=test_size,
            random_state=random_state,
            class_names=[str(c) for c in unique_classes],
        )

        return JSONResponse(content={
            "success":        True,
            "message":        f"{model_name.upper()} trained and evaluated successfully.",
            "model_name":     metrics["model_name"],
            "accuracy":       metrics["accuracy"],
            "precision_macro":metrics["precision_macro"],
            "recall_macro":   metrics["recall_macro"],
            "f1_macro":       metrics["f1_macro"],
            "f1_weighted":    metrics["f1_weighted"],
            "confusion_matrix":       metrics["confusion_matrix"],
            "classification_report":  metrics["classification_report"],
            "train_samples":  metrics["train_samples"],
            "test_samples":   metrics["test_samples"],
            "n_classes":      metrics["n_classes"],
            "class_names":    [str(c) for c in unique_classes],
            "feature_matrix_shape": list(X.shape),
            "epoch_info": {
                "n_epochs":        n_epochs,
                "window_size_sec": window_size_sec,
                "overlap_ratio":   overlap_ratio,
                "sampling_rate":   sampling_rate,
            },
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model training failed: {str(e)}"
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: POST /api/cross-validate
# Phase 3 — Group K-Fold Cross-Validation (Subject-Independent Evaluation)
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/cross-validate")
async def cross_validate_endpoint(
    file: UploadFile = File(..., description="EEG dataset CSV file (with event_label and subject_id columns)"),
    model_name: str  = Form("svm",  description="Classifier: 'svm' | 'random_forest' | 'xgboost'"),
    window_size_sec: float = Form(2.0, description="Epoch window size in seconds"),
    overlap_ratio:   float = Form(0.5, description="Sliding window overlap ratio [0, 1)"),
    n_splits:        int   = Form(5,   description="Number of folds (partitions)"),
    random_state:    int   = Form(42,  description="Random seed for model reproducibility"),
):
    """
    Perform Group K-Fold Cross-Validation on the uploaded EEG dataset.

    Groups epochs by subject_id (from the CSV dataset) to partition train/test sets,
    preventing subject-level data leakage. Capping n_splits automatically if n_splits
    exceeds unique subjects.
    """
    # ── Parameter validation ──────────────────────────────────────────────────
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    if model_name not in CLASSIFIERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'. Choose from: {CLASSIFIERS}"
        )

    if window_size_sec <= 0:
        raise HTTPException(status_code=400, detail="window_size_sec must be greater than 0.")

    if not (0.0 <= overlap_ratio < 1.0):
        raise HTTPException(status_code=400, detail="overlap_ratio must be in [0.0, 1.0).")

    if n_splits < 2:
        raise HTTPException(status_code=400, detail="n_splits must be at least 2.")

    temp_file_path = None
    try:
        # ── Save uploaded CSV to temp file ─────────────────────────────────
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_file_path = tmp.name

        # ── Load & validate EEG data ────────────────────────────────────────
        loader = EEGDataLoader(temp_file_path)
        df, metadata = loader.load_and_validate()

        # Check if subject_id column exists
        if "subject_id" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="Column 'subject_id' not found in dataset. Group K-Fold cross-validation requires a 'subject_id' column to partition train/test splits without leakage."
            )

        sampling_rate = metadata.get("sampling_rate_hz", metadata.get("sampling_rate", 250))
        eeg_channels  = metadata.get("channels", metadata.get("eeg_channels", []))

        if not eeg_channels:
            raise HTTPException(status_code=400, detail="No EEG channels found in the uploaded CSV.")

        # ── Epoch segmentation ──────────────────────────────────────────────
        epoch_result = create_epochs_from_dataframe(
            df=df,
            channels=eeg_channels,
            fs=sampling_rate,
            window_size_sec=window_size_sec,
            overlap_ratio=overlap_ratio,
        )

        epochs_array = epoch_result["epochs"]    # shape: (n_epochs, n_channels, samples)
        labels_list  = epoch_result.get("labels", None)
        subjects_list = epoch_result.get("subjects", None)
        n_epochs     = epoch_result["n_epochs"]

        if n_epochs == 0:
            raise HTTPException(status_code=400, detail="No epochs could be created. Check window_size_sec and data length.")

        if subjects_list is None or len(subjects_list) == 0:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract subject groupings from epochs. Check 'subject_id' values."
            )

        # ── Feature extraction ──────────────────────────────────────────────
        feature_df = extract_all_features(
            epochs=epochs_array,
            fs=sampling_rate,
            channel_names=eeg_channels,
            labels=labels_list,
        )
        feature_names = [c for c in feature_df.columns if c not in ("event", "event_label", "subject_id", "subject")]

        # ── Prepare X (features) and y (labels) ─────────────────────────────
        label_col = "event" if "event" in feature_df.columns else "event_label"
        if label_col not in feature_df.columns:
            if labels_list is not None:
                feature_df["event"] = labels_list
                label_col = "event"
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No event label column found. The CSV must contain an event_label column."
                )

        X = feature_df[feature_names].values.astype(np.float32)
        y = feature_df[label_col].values

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"At least 2 event classes required. Found: {unique_classes.tolist()}"
            )

        # ── Run Group K-Fold Cross-Validation ───────────────────────────────
        cv_results = cross_validate_model(
            X=X,
            y=y,
            groups=np.array(subjects_list),
            model_name=model_name,
            n_splits=n_splits,
            random_state=random_state,
            class_names=[str(c) for c in unique_classes],
        )

        return JSONResponse(content={
            "success":                      True,
            "message":                      f"Group K-Fold Cross-Validation for {model_name.upper()} completed.",
            "model_name":                   cv_results["model_name"],
            "n_splits":                     cv_results["n_splits"],
            "mean_accuracy":                cv_results["mean_accuracy"],
            "std_accuracy":                 cv_results["std_accuracy"],
            "mean_precision_macro":         cv_results["mean_precision_macro"],
            "mean_recall_macro":            cv_results["mean_recall_macro"],
            "mean_f1_macro":                cv_results["mean_f1_macro"],
            "accumulated_confusion_matrix": cv_results["accumulated_confusion_matrix"],
            "folds":                        cv_results["folds"],
            "n_samples":                    cv_results["n_samples"],
            "n_classes":                    cv_results["n_classes"],
            "class_names":                  cv_results["class_names"],
            "feature_matrix_shape":         list(X.shape),
            "epoch_info": {
                "n_epochs":        n_epochs,
                "window_size_sec": window_size_sec,
                "overlap_ratio":   overlap_ratio,
                "sampling_rate":   sampling_rate,
            },
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cross-validation pipeline failed: {str(e)}"
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

