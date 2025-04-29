import sys
import os
import whisper

# Dictionary to cache loaded models by (model_name, download_root)
_loaded_whisper_models = {}

def _is_running_in_bundle():
    """Check if the application is running in a bundled environment (e.g., PyInstaller)"""
    return getattr(sys, 'frozen', False)

def _get_correct_path(relative_path):
    """Get the absolute path to resource, works for both dev and PyInstaller environments"""
    try:
        base_path = sys._MEIPASS if _is_running_in_bundle() else os.path.dirname(os.path.abspath(__file__))
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

def get_whisper_model(model_name='base', download_root=None):
    """
    Loads and returns a singleton Whisper model instance for the given model_name and download_root.
    If already loaded, returns the cached instance.
    """
    global _loaded_whisper_models
    key = (model_name, download_root)
    
    if key not in _loaded_whisper_models:
        is_bundled = _is_running_in_bundle()
        
        if is_bundled and download_root:
            print(f"Running in bundled environment, loading Whisper model '{model_name}' from '{download_root}'")
        elif download_root:
            print(f"Loading Whisper model '{model_name}' from custom path '{download_root}'")
        else:
            print(f"Loading Whisper model '{model_name}' from default location (will download if needed)")
            
        _loaded_whisper_models[key] = whisper.load_model(model_name, download_root=download_root)
        
    return _loaded_whisper_models[key] 