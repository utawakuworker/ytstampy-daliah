import sys
import os
from faster_whisper import WhisperModel

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

def get_whisper_model(model_name='base', download_root=None, device="cpu", compute_type="int8"):
    """
    Loads and returns a singleton faster-whisper model instance.

    Args:
        model_name (str): The name of the Whisper model (e.g., 'base', 'small').
                          Corresponds to models compatible with faster-whisper.
        download_root (str, optional): Path to download/load models. Faster-whisper uses
                                       its own caching, so this might be less critical unless
                                       bundling. Defaults to None (uses faster-whisper default).
        device (str): Device to load the model on ("cpu", "cuda").
        compute_type (str): Type for computation ("float16", "int8", etc.).
    """
    global _loaded_whisper_models

    # Create a unique key based on model name, device, and compute type
    key = (model_name, device, compute_type)

    if key not in _loaded_whisper_models:
        print(f"Loading faster-whisper model: {model_name} (Device: {device}, Compute: {compute_type})")
        # Note: faster-whisper handles its own download/caching.
        # The 'download_root' from the original whisper library might not directly map.
        # If you need to force a specific *local* directory containing a pre-converted model,
        # you would pass that path directly as the model_name/identifier.
        # For now, we rely on faster-whisper's ability to fetch standard models.
        try:
            # If download_root is specified AND exists, treat it as the model identifier path
            if download_root and os.path.isdir(download_root):
                 model_path = os.path.join(download_root, f"faster-whisper-{model_name}") # Assuming a structure
                 print(f"Attempting to load model from local path: {model_path}")
                 # Check if the specific model directory exists within the download_root
                 if not os.path.isdir(model_path):
                      print(f"Warning: Directory {model_path} not found. Falling back to default download.")
                      model_path = model_name # Fallback to model name for automatic download
            else:
                 model_path = model_name # Use model name for automatic download

            _loaded_whisper_models[key] = WhisperModel(
                model_path, # Use determined path or model name
                device=device,
                compute_type=compute_type,
                # download_root=download_root # download_root is not a direct parameter for WhisperModel constructor
            )
            print(f"Model {model_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading faster-whisper model {model_name}: {e}")
            # Consider re-raising or returning None depending on desired error handling
            raise # Re-raise the exception to indicate failure
    else:
        print(f"Returning existing faster-whisper model instance for: {key}")

    return _loaded_whisper_models[key]

def clear_loaded_models():
    """ Clears the cache of loaded models. """
    global _loaded_whisper_models
    _loaded_whisper_models = {}
    print("Cleared cached faster-whisper models.") 