"""
Audio processing module for singing detection.
"""
# Import main classes for easier access
from .loader import AudioLoader, LocalAudioLoader, YouTubeAudioLoader, AudioLoaderFactory
from .feature_extraction import (
    FeatureExtractor, 
    SpectralFeatureExtractor, 
    MFCCFeatureExtractor, 
    HarmonicFeatureExtractor,
    PitchFeatureExtractor,
    FeatureExtractorFacade
)
from .utils import format_timestamp, play_audio_segment 