import whisper

# Dictionary to cache loaded models by (model_name, download_root)
_loaded_whisper_models = {}

def get_whisper_model(model_name='base', download_root=None):
    """
    Loads and returns a singleton Whisper model instance for the given model_name and download_root.
    If already loaded, returns the cached instance.
    """
    global _loaded_whisper_models
    key = (model_name, download_root)
    if key not in _loaded_whisper_models:
        print("Not running in bundle, using default Whisper model loading.")
        print(f"Loading Whisper model '{model_name}' with download_root='{download_root}'")
        _loaded_whisper_models[key] = whisper.load_model(model_name, download_root=download_root)
    return _loaded_whisper_models[key] 