# Default configuration settings for the analysis pipeline

DEFAULT_CONFIG = {
    # Input: Default to None, user must provide via GUI
    'url': None,
    'file': None,

    # Reference segments (provide representative time points in seconds)
    'singing_ref_time': 1250,      # Time point with clear singing
    'non_singing_ref_time': 230,   # Time point with clear non-singing/instrumental
    'ref_duration': 2.0,         # Duration (in seconds) for reference segments

    # Detection Parameters
    'hmm_threshold': 0.55,
    'min_segment_duration': 10.0, # Min duration for HMM detection AND final filtering
    'min_segment_gap': 1.5,       # Min gap for HMM detection
    'dim_reduction': 'pca',
    'n_components': 4,

    # Feature Extraction
    'enable_hpss': False,          # Enable Harmonic-Percussive Source Separation

    # Identification Parameters
    'min_duration_for_id': 30.0, # Min duration of a segment to attempt identification
    'max_duration_for_id': 60,
    'whisper_model': 'base',     # Whisper model size
    'gemini_api_key': '',        # IMPORTANT: Best loaded from env var or secure storage in a real app

    # Output Settings
    'output_dir': './output_analysis_gui', # Default output dir for GUI runs
    'visualize': False, # Default setting for GUI checkbox
    'save_results_json': True, # Default setting for GUI checkbox
    'save_results_dataframe': True, # Default setting for GUI checkbox
    'results_json_file': 'song_identification_results.json',
    'results_dataframe_file': 'song_identification_results.csv'
}
