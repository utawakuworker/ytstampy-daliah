import threading
import os
from typing import Dict, Any, Optional, Callable, List

# Import the pipeline and config from the model package
try:
    from model.analysis_pipeline import SingingAnalysisPipeline
    from model.config import DEFAULT_CONFIG
except ImportError:
    # Handle potential path issues if running viewmodel directly
    import sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from model.analysis_pipeline import SingingAnalysisPipeline
    from model.config import DEFAULT_CONFIG


class AnalysisViewModel:
    """
    Manages the state and logic for the Singing Analysis GUI.
    Acts as the intermediary between the View (GUI) and the Model (Pipeline).
    """
    def __init__(self, view_update_callback: Callable):
        """
        Args:
            view_update_callback: A function the ViewModel calls to signal
                                  the View that its state has changed and
                                  the UI needs refreshing.
        """
        self.view_update_callback = view_update_callback

        # --- UI State Properties ---
        # Configuration state (initialized from defaults, updated by View)
        self.config: Dict[str, Any] = DEFAULT_CONFIG.copy()

        # Runtime state
        self.is_running: bool = False
        self.status_message: str = "Idle"
        self.progress_value: Optional[float] = 0.0
        self.log_messages: List[str] = ["[INFO] Application started."]

        # Results state
        self.analysis_results: Optional[Dict[str, Any]] = None

        # Internal state
        self.pipeline_thread: Optional[threading.Thread] = None
        self.pipeline_instance: Optional[SingingAnalysisPipeline] = None # Hold instance for save/vis

    def _notify_view(self):
        """Signals the view to update itself based on current ViewModel state."""
        # In a real Flet app, this triggers the page/control updates
        self.view_update_callback()

    # --- Status/Progress Callbacks for the Model ---
    # These methods are passed to the pipeline instance and called by it

    def _pipeline_status_update(self, message: str):
        """Receives status updates from the pipeline."""
        self.status_message = message
        self.log_messages.append(f"[STATUS] {message}")
        # Limit log size
        if len(self.log_messages) > 200:
            self.log_messages = self.log_messages[-200:]
        self._notify_view() # Tell the view to refresh

    def _pipeline_progress_update(self, value: Optional[float]):
        """Receives progress updates from the pipeline."""
        self.progress_value = value
        self._notify_view() # Tell the view to refresh

    # --- Actions Triggered by the View ---

    def update_config_value(self, key: str, value: Any):
        """Updates a single configuration value."""
        # Add validation/type conversion as needed
        print(f"ViewModel: Updating config '{key}' to '{value}'")
        self.config[key] = value
        # No need to notify view unless the config value itself is displayed directly

    def update_full_config(self, new_config_values: Dict[str, Any]):
        """Updates the entire configuration dictionary (e.g., from GUI fields)."""
        # Perform validation if necessary
        self.config.update(new_config_values)
        print(f"ViewModel: Full config updated.")
        # No need to notify view

    def start_analysis(self):
        """Starts the analysis process in a background thread."""
        if self.is_running:
            self._pipeline_status_update("Analysis is already running.")
            return

        self.is_running = True
        self.analysis_results = None # Clear previous results
        self.log_messages = ["[INFO] Starting analysis..."]
        self.status_message = "Initializing..."
        self.progress_value = 0.0
        self._notify_view() # Update UI immediately (show running state, clear logs/results)

        # Create pipeline instance with callbacks
        # Make sure API key is handled (e.g., loaded from env if empty in config)
        current_config = self.config.copy()
        if not current_config.get('gemini_api_key'):
             current_config['gemini_api_key'] = os.environ.get('GEMINI_API_KEY', '')
             if not current_config['gemini_api_key']:
                  self._pipeline_status_update("Warning: Gemini API Key not set in config or environment.")
                  # Continue without API key, identification will be skipped by pipeline

        self.pipeline_instance = SingingAnalysisPipeline(
            config=current_config,
            status_callback=self._pipeline_status_update,
            progress_callback=self._pipeline_progress_update
        )

        # Define the target function for the thread
        def run_in_thread():
            results = None
            try:
                results = self.pipeline_instance.run() # Run the analysis
                self.analysis_results = results # Store results
            except Exception as e:
                self._pipeline_status_update(f"Pipeline Thread Error: {e}")
                import traceback
                self.log_messages.append(f"[ERROR] {traceback.format_exc()}")
            finally:
                self.is_running = False
                # Ensure final state update reaches the view
                # Status message should be set by the pipeline on success/failure
                self._notify_view()

        # Start the thread
        self.pipeline_thread = threading.Thread(target=run_in_thread, daemon=True)
        self.pipeline_thread.start()

    def save_results(self):
        """Triggers saving results using the completed pipeline instance."""
        if self.is_running:
            self._pipeline_status_update("Cannot save results: Analysis is running.")
            return
        if not self.pipeline_instance or not self.analysis_results:
            self._pipeline_status_update("Cannot save results: No analysis has been run successfully.")
            return

        # Run saving in a separate thread? Usually fast enough, but consider if saving is slow.
        try:
            self._pipeline_status_update("Saving results...")
            self.pipeline_instance.save_results() # Uses callbacks internally now
            # Status will be updated by the save_results method via callback
        except Exception as e:
             self._pipeline_status_update(f"Error triggering save: {e}")
        finally:
             self._notify_view() # Ensure UI reflects final status

    def visualize_results(self):
        """Triggers visualization using the completed pipeline instance."""
        if self.is_running:
            self._pipeline_status_update("Cannot visualize: Analysis is running.")
            return
        if not self.pipeline_instance or not self.analysis_results:
            self._pipeline_status_update("Cannot visualize: No analysis has been run successfully.")
            return
        if not self.config.get('visualize'):
             self._pipeline_status_update("Visualization is disabled in configuration.")
             return

        # IMPORTANT: Matplotlib's plt.show() blocks and might have issues
        # outside the main thread depending on the backend.
        # Running this in a separate thread is often problematic.
        # Simplest: Run directly, acknowledging it might block the *ViewModel* briefly
        # (but not the GUI if called correctly).
        # Better: Modify plotting functions to save to file or return figures,
        # then display in Flet.
        try:
            self._pipeline_status_update("Generating visualization (may block)...")
            self.pipeline_instance.visualize() # Uses callbacks internally now
            # Status will be updated by the visualize method via callback
        except Exception as e:
             self._pipeline_status_update(f"Error triggering visualization: {e}")
        finally:
             self._notify_view() # Ensure UI reflects final status


    # --- Data Access for the View ---
    # View calls these methods to get the data it needs to display

    def get_status_message(self) -> str:
        return self.status_message

    def get_progress_value(self) -> Optional[float]:
        return self.progress_value

    def get_log_messages(self) -> list[str]:
        # Return a copy to prevent accidental modification by the view
        return list(self.log_messages)

    def get_detected_segments(self) -> list:
        return self.analysis_results.get('final_segments', []) if self.analysis_results else []

    def get_identification_results(self) -> list:
         # This contains results only for segments that were long enough for ID attempt
         return self.analysis_results.get('identification_results', []) if self.analysis_results else []

    def get_summary_info(self) -> Dict[str, Any]:
         if not self.analysis_results or not self.analysis_results.get('final_segments'):
             return {"text": "No analysis performed yet.", "count": 0}
         segments = self.analysis_results['final_segments']
         if not segments:
              return {"text": "No singing segments detected.", "count": 0}

         total_duration = self.analysis_results.get('total_duration', 0)
         total_singing_time = sum(end - start for start, end in segments)
         percentage = (total_singing_time / total_duration * 100) if total_duration else 0
         return {
             "text": f"Total Singing: {total_singing_time:.1f}s ({percentage:.1f}%)",
             "count": len(segments)
         }

    def is_analysis_running(self) -> bool:
        return self.is_running

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Safely get a config value for the View to display."""
        return self.config.get(key, default)

    def get_full_config(self) -> Dict[str, Any]:
        """Returns a copy of the current configuration."""
        return self.config.copy() 