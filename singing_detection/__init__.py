"""
Singing Detection System
A SOLID-based system for detecting singing segments in audio.
"""

# Audio Processing
from .audio.loader import AudioLoaderFactory, AudioLoader, LocalAudioLoader, YouTubeAudioLoader
from .audio.feature_extraction import FeatureExtractorFacade

# Detection
from .detection.detection_engine import (
    DetectionEngine, 
    HMMDetectionEngine,
    ClusterEnhancedHMMDetectionEngine
)

# Segment Processing
from .segments.segment_processor import (
    SegmentProcessor,
    SegmentValidator,
    SegmentFilter,
    SegmentRefiner,
    SegmentMerger,
    SegmentProcessingPipeline
)

# Visualization
from .visualization.plots import plot_waveform_with_segments, plot_feature_comparison

__version__ = '2.0.0'  # New major version to indicate breaking changes 