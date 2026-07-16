import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class EEGDataLoader:
    """
    A utility class for loading, validating, and cleaning EEG data from CSV files.
    Identifies channel columns, detects sampling frequency, and handles missing values.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df: Optional[pd.DataFrame] = None
        self.channels: List[str] = []
        self.time_col: Optional[str] = None
        self.subject_col: Optional[str] = None
        self.event_col: Optional[str] = None
        self.fs: float = 250.0  # Default fallback sampling rate in Hz
        self.info: Dict[str, Any] = {}

    def load_and_validate(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads the CSV file, validates its structure, cleans missing values,
        and extracts key metadata such as channel names and sampling frequency.
        """
        try:
            # 1. Load CSV file
            self.df = pd.read_csv(self.file_path)
            if self.df.empty:
                raise ValueError("The uploaded CSV file is empty.")
            
            # Remove any fully empty columns or rows
            self.df.dropna(how='all', inplace=True)
            self.df.dropna(how='all', axis=1, inplace=True)

            # 2. Identify and classify columns
            self._identify_columns()

            # 3. Clean missing values (NaNs) in EEG channels
            self._clean_missing_values()

            # 4. Estimate sampling frequency (fs)
            self._estimate_sampling_rate()

            # 5. Compile metadata info
            self.info = {
                "file_name": self.file_path.split("/")[-1].split("\\")[-1],
                "num_samples": len(self.df),
                "num_channels": len(self.channels),
                "channels": self.channels,
                "sampling_rate_hz": round(self.fs, 2),
                "time_duration_sec": round((len(self.df) - 1) / self.fs,2) if len(self.df) > 1 else 0.0,
                "has_subject_id": self.subject_col is not None,
                "has_events": self.event_col is not None,
                "subject_ids": self.df[self.subject_col].unique().tolist() if self.subject_col else [],
                "event_types": self.df[self.event_col].unique().tolist() if self.event_col else []
            }

            return self.df, self.info

        except Exception as e:
            raise ValueError(f"Failed to process EEG CSV file: {str(e)}")

    def _identify_columns(self):
        """
        Scans columns and classifies them into Time, Subject, Event/Label, or EEG Channels.
        Non-EEG metadata columns are identified by common keyword patterns.
        """
        if self.df is None:
            return

        columns = list(self.df.columns)
        metadata_cols = []

        # Identify Time column
        time_keywords = ['time', 'timestamp', 'sec', 'ms']
        for col in columns:
            if any(key in col.lower() for key in time_keywords):
                self.time_col = col
                metadata_cols.append(col)
                break

        # Identify Subject column
        subject_keywords = {
            "subject",
            "subject_id",
            "participant",
            "participant_id",
            "person",
            "person_id"
        }
        for col in columns:
            if (col not in metadata_cols and col.lower().strip() in subject_keywords):
                self.subject_col = col
                metadata_cols.append(col)
                break

        # Identify Event/Label column
        event_keywords = ['event', 'label', 'state', 'target', 'class', 'marker']
        for col in columns:
            if col not in metadata_cols and any(key in col.lower() for key in event_keywords):
                self.event_col = col
                metadata_cols.append(col)
                break
            

        # Treat all remaining numeric columns as EEG channels
        numeric_metadata_keywords = [
            'age',
            'trial',
            'trial_id',
            'session',
            'session_id',
            'recording_id',
            'sample_id',
            'sampling_rate',
            'fs'
        ]
        self.channels = []
        for col in columns:
            normalized_col = col.lower().strip()

            if col not in metadata_cols:
                if normalized_col in numeric_metadata_keywords:
                    metadata_cols.append(col)
                elif pd.api.types.is_numeric_dtype(self.df[col]):
                    self.channels.append(col)
                else:
                    metadata_cols.append(col)

        if not self.channels:
            raise ValueError("No numeric EEG channel columns detected in the CSV file.")

    def _clean_missing_values(self):
        """
        Checks for missing values (NaNs) in EEG channels and cleans them.
        Linear interpolation is used for continuous time-series data to prevent artifacts,
        falling back to forward/backward fill if interpolation is not possible.
        """
        if self.df is None:
            return

        # Check for NaNs across channel columns
        nan_counts = self.df[self.channels].isna().sum()
        total_nans = nan_counts.sum()

        if total_nans > 0:
            print(f"Warning: Detected {total_nans} missing values in EEG channels. Performing interpolation...")
            # Interpolate channel columns linearly
            if self.subject_col:
                self.df[self.channels] = (
                    self.df.groupby(self.subject_col)[self.channels]
                    .transform(
                        lambda group: group.interpolate(
                            method='linear',
                            limit=5,
                            limit_direction='both'
                        )
                    )
                )
            else:
                self.df[self.channels] = (
                    self.df[self.channels]
                    .interpolate(
                        method='linear',
                        limit=5,
                        limit_direction='both'
                    )
                )
            
            # If any NaNs remain (e.g. at the edges), fill with forward/backward fill
            if self.df[self.channels].isna().sum().sum() > 0:
                if self.subject_col:
                    self.df[self.channels] = (
                        self.df.groupby(self.subject_col)[self.channels]
                        .transform(lambda group: group.ffill().bfill())
                    )
                else:
                    self.df[self.channels] = (
                        self.df[self.channels]
                        .ffill()
                        .bfill()
                    )

    def _estimate_sampling_rate(self):
        """
        Estimates the sampling frequency (fs) in Hz.
        Calculates the mean time difference (dt) between consecutive samples.
        """
        if self.df is None:
            return

        if self.time_col and len(self.df) > 1:
            try:
                # Calculate diffs
                time_diffs = np.diff(self.df[self.time_col].values)
                valid_diffs = time_diffs[
                    np.isfinite(time_diffs) & (time_diffs > 0)
                ]

                if len(valid_diffs) == 0:
                    self.fs = 250.0
                    return

                median_dt = np.median(valid_diffs)

                if median_dt > 0:
                    # If time is in milliseconds (common in EEG), convert to seconds
                    # We assume ms if the mean difference is large (e.g. > 0.05 for 250Hz is 4ms)
                    # Standard EEG fs is usually 100Hz - 1000Hz, so dt is 1ms - 10ms.
                    # If mean dt is > 0.5, we assume ms if timestamps are integers or large.
                    # Let's check magnitude of the first timestamp
                    
                    if median_dt >= 1.0:  # Time is likely in milliseconds (e.g., 4ms, 10ms diffs)
                        self.fs = 1000.0 / median_dt
                    else:  # Time is likely in seconds (e.g., 0.004s, 0.01s diffs)
                        self.fs = 1.0 / median_dt
                
                # Check for extreme outlier sampling rates
                if self.fs <= 0 or self.fs > 10000:
                    self.fs = 250.0  # Fallback to standard 250 Hz
            except Exception:
                self.fs = 250.0  # Fallback on calculation failure
        else:
            self.fs = 250.0  # Fallback if time column is missing
