"""
Audio processing module for singing detection.
"""
# Import main classes for easier access
from singing_detection.audio.loader import AudioLoader, LocalAudioLoader, YouTubeAudioLoader, AudioLoaderFactory
from singing_detection.audio.feature_extraction import (
    FeatureExtractor, 
    SpectralFeatureExtractor, 
    MFCCFeatureExtractor, 
    HarmonicFeatureExtractor,
    PitchFeatureExtractor,
    FeatureExtractorFacade
)
from singing_detection.audio.utils import format_timestamp, export_results_as_video_chapters