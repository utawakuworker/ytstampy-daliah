import flet as ft
import sys
import os

# Ensure the current directory is in the path to find sibling packages
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from gui.app_view import AppView
    from viewmodel.analysis_viewmodel import AnalysisViewModel
except ImportError as e:
     print(f"Error importing View/ViewModel in main.py: {e}")
     print("Ensure gui/ and viewmodel/ packages are correctly structured.")
     sys.exit(1)

def run_app(page: ft.Page):
    """Configures and runs the Flet application."""

    # --- Define the callback for the ViewModel ---
    # This function will be called by the ViewModel to trigger UI updates.
    # In Flet, the most straightforward way is often to just call page.update()
    # after the AppView's internal update method runs.
    # So, we pass AppView's update method itself, or a wrapper.
    # We'll instantiate AppView first, then pass its update method.

    app_view_instance = None # Placeholder

    def trigger_view_update():
        """Callback function passed to ViewModel."""
        if app_view_instance:
            # The AppView's update_view_state method should handle
            # updating controls and calling page.update() itself.
            app_view_instance.update_view_state()
        else:
            # Fallback if called before view is ready (shouldn't happen often)
            page.update()


    # 1. Create ViewModel instance
    view_model = AnalysisViewModel(view_update_callback=trigger_view_update)

    # 2. Create View instance, passing the ViewModel
    app_view_instance = AppView(view_model) # Now we have the instance

    # 3. Add the main view control to the page
    page.add(app_view_instance)

    # Initial page setup
    page.title = "Singing Analysis Tool"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.window_width = 850 # Adjusted size
    page.window_height = 750
    page.update() # Initial render

if __name__ == "__main__":
    # Set environment variable for API key (example - better to set externally)
    # os.environ['GEMINI_API_KEY'] = 'YOUR_ACTUAL_API_KEY'

    ft.app(target=run_app) 