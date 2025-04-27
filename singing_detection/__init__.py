"""
Singing Detection System
A SOLID-based system for detecting singing segments in audio.
"""

# Audio Processing
from singing_detection.audio.loader import AudioLoaderFactory, AudioLoader, LocalAudioLoader, YouTubeAudioLoader
from singing_detection.audio.feature_extraction import FeatureExtractorFacade

# Detection
from singing_detection.detection import (
    DetectionEngine,
    FeatureEngineer,
    InterludeAnalyzer,
    SingingDetectionPipeline,


# Segment Processing
from singing_detection.segments.segment_processor import (
    SegmentProcessor,
    SegmentValidator,
    SegmentFilter,
    SegmentRefiner,
    SegmentMerger,
    SegmentProcessingPipeline
)

# Visualization
from singing_detection.visualization.plots import plot_waveform_with_segments, plot_feature_comparison

__version__ = '2.0.0'  # New major version to indicate breaking changes 