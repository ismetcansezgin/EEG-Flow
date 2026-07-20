import os
import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.utils.data_loader import EEGDataLoader

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
