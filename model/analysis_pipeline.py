import re
import warnings
from typing import Any, Callable, Dict, List, Optional


from model.pipeline_steps import PipelineStep

# warnings.filterwarnings('ignore')

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
    Also manages the stop request flag.
    """
    def __init__(self, status_callback: Optional[Callable[[str], None]] = None, progress_callback: Optional[Callable[[Optional[float]], None]] = None):
        self.status_callback = status_callback or (lambda msg: None)
        self.progress_callback = progress_callback or (lambda val: None)
        self._stop_requested = False

    def status(self, message: str):
        if not self._stop_requested:
             self.status_callback(message)

    def progress(self, value: Optional[float], message: Optional[str] = None):
        if not self._stop_requested:
            if message:
                self.status(message)
            self.progress_callback(value)

    def request_stop(self):
        """Sets the stop requested flag to True."""
        self._stop_requested = True
        self.status_callback("Stop requested by user.")

    def should_stop(self) -> bool:
        """Checks if a stop has been requested."""
        return self._stop_requested
    
    def reset_stop_flag(self):
        """Resets the stop requested flag to False."""
        self._stop_requested = False

class SingingAnalysisOrchestrator:
    """
    Orchestrates the execution of a list of PipelineStep instances, managing context and reporting.
    """
    def __init__(self, steps: List[PipelineStep], reporter: Optional[StatusReporter] = None):
        self.steps = steps
        self.reporter = reporter or StatusReporter()
        self.context: Dict[str, Any] = {}

    def request_stop(self):
        self.reporter.request_stop()

    def run(self) -> Optional[Dict[str, Any]]:
        self.reporter.reset_stop_flag()
        self.reporter.progress(0.0, "Starting analysis pipeline...")
        for idx, step in enumerate(self.steps):
            if self.reporter.should_stop():
                return None
            step_name = step.__class__.__name__
            self.reporter.status(f"Running step: {step_name}")
            step_success = False
            try:
                step_success = step.run(self.context)
            except Exception as e:
                error = f"Error during {step_name}: {e}"
                self.reporter.status(error)
                self.context['error'] = error
                return None
            
            if not step_success:
                error = self.context.get('error', f'Step {step_name} returned False')
                self.reporter.status(f"Step {step_name} failed: {error}")
                return None
            if self.reporter.should_stop():
                 return None
            self.reporter.progress((idx + 1) / len(self.steps))
            
        if self.reporter.should_stop():
             return None
             
        self.reporter.progress(1.0, "Analysis pipeline completed.")
        return self.context 