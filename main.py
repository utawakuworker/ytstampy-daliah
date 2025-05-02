import os
import sys

import flet as ft

# Ensure the current directory is in the path to find sibling packages
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from gui.app_view import AppView
    from gui.localization import LocalizationManager
    from viewmodel.analysis_viewmodel import AnalysisViewModel
except ImportError as e:
     print(f"Error importing View/ViewModel/Localization in main.py: {e}")
     print("Ensure gui/ and viewmodel/ packages are correctly structured.")
     sys.exit(1)

# --- Initialize Localization (remains global or in scope) ---
print("Initializing LocalizationManager...")
loc_manager = LocalizationManager(locales_dir="locales", default_lang="en")
print(f"LocalizationManager initialized. Current lang: {loc_manager.get_current_language()}")
available_langs = loc_manager.get_available_languages()
print(f"Available languages found: {available_langs}")

def run_app(page: ft.Page):
    """Configures and runs the Flet application."""
    print("--- run_app called ---")

    # Set window size for desktop apps using the window property
    page.window.width = 600
    page.window.height = 1000
    # Optionally, make the window not resizable:
    # page.window.resizable = False
    page.title = "UtaStamper"  # Use fixed app name without translation
    page.update()

    # --- Theme Check (Comment out if you have custom themes) ---
    # page.theme_mode = ft.ThemeMode.LIGHT # Or DARK / SYSTEM
    # page.theme = ft.Theme(...)
    # page.dark_theme = ft.Theme(...)
    # --- End Theme Check ---

    # --- Keep ViewModel instance persistent within run_app scope ---
    # Pass a lambda that calls page.update directly for simplicity here.
    # The view_model's callback will be properly assigned within build_and_set_view
    view_model = AnalysisViewModel(view_update_callback=lambda: page.update())
    print("ViewModel created.")

    # ---> Explicit Update after AppBar assignment <---
    print("Attempting explicit page update after AppBar assignment...")
    page.update()
    print("Explicit update called.")

    # --- Function to build/rebuild the main AppView ---
    def build_and_set_view():
        # Get the current translator function from the manager
        current_tr = loc_manager.tr
        print(f"--- build_and_set_view called (Lang: {loc_manager.get_current_language()}) ---")
        print(f"Building view with language: {loc_manager.get_current_language()}") # Debug print
        # Create the AppView instance
        app_view_instance = AppView(view_model=view_model, tr=current_tr)
        # Assign the view's update method to the view model's callback
        # This links the ViewModel back to the *current* view instance
        view_model.view_update_callback = app_view_instance.update_view_state

        # Update page content (replace views)
        page.views.clear()
        page.views.append(app_view_instance)

        print("--- build_and_set_view finished ---")

    # --- Initial View Setup ---
    build_and_set_view() # Call it once to build the initial view

    # --- Page Settings ---
    page.vertical_alignment = ft.MainAxisAlignment.START
    print("Page settings applied. Calling final page.update()...")
    page.update() # Final initial render
    print("--- run_app finished ---")

if __name__ == "__main__":

    print("Starting Flet app...")
    ft.app(target=run_app, view=ft.FLET_APP) 