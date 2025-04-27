import os
import re
import json
import warnings
from typing import Dict, Any, Optional, List, Callable
from urllib.parse import urlparse, parse_qs
import pandas as pd

from model.data_models import AnalysisResults, Segment, SegmentIdentification, SongIdentificationResult
from model.pipeline_steps import PipelineStep

warnings.filterwarnings('ignore')

try:
    from singing_detection.audio.loader import AudioLoaderFactory
    from singing_detection.audio.feature_extraction import FeatureExtractorFacade
    from singing_detection.detection.detection_engine import ClusterEnhancedHMMDetectionEngine
    from singing_detection.segments.segment_processor import SegmentFilter, SegmentProcessingPipeline, SegmentRefiner
    from singing_detection.visualization.plots import plot_waveform_with_segments, plot_feature_comparison
    from singing_detection.identification.song_identifier import SongIdentifier
except ImportError as e:
    print(f"Error importing project modules in analysis_pipeline.py: {e}")
    raise

# --- Helper Function ---
def sanitize_filename(name: str, max_length: int = 100) -> str:
    """Removes or replaces invalid filename characters and limits length."""
    if not name:
        return "unknown_source"
    sanitized = re.sub(r'[^a-zA-Z0-9_.-]+', '_', name)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_.')
    return sanitized[:max_length] if len(sanitized) > max_length else sanitized

class StatusReporter:
    """
    Handles status and progress reporting for the pipeline.
    """
    def __init__(self, status_callback: Optional[Callable[[str], None]] = None, progress_callback: Optional[Callable[[Optional[float]], None]] = None):
        self.status_callback = status_callback or (lambda msg: None)
        self.progress_callback = progress_callback or (lambda val: None)

    def status(self, message: str):
        self.status_callback(message)

    def progress(self, value: Optional[float], message: Optional[str] = None):
        if message:
            self.status(message)
        self.progress_callback(value)

class SingingAnalysisOrchestrator:
    """
    Orchestrates the execution of a list of PipelineStep instances, managing context and reporting.
    """
    def __init__(self, steps: List[PipelineStep], reporter: Optional[StatusReporter] = None):
        self.steps = steps
        self.reporter = reporter or StatusReporter()
        self.context: Dict[str, Any] = {}
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True
        self.reporter.status("Stop requested. Pipeline will halt after the current step.")

    def run(self) -> Optional[Dict[str, Any]]:
        self.reporter.progress(0.0, "Starting analysis pipeline...")
        for idx, step in enumerate(self.steps):
            if self._stop_requested:
                self.reporter.status("Pipeline stopped by user.")
                return None
            self.reporter.status(f"Running step: {step.__class__.__name__}")
            if not step.run(self.context):
                error = self.context.get('error', 'Unknown error')
                self.reporter.status(f"Step {step.__class__.__name__} failed: {error}")
                return None
            self.reporter.progress((idx + 1) / len(self.steps))
        self.reporter.progress(1.0, "Analysis pipeline completed.")
        return self.context 