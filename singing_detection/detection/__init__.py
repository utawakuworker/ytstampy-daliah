"""
Detection algorithms module for singing detection.
"""
from singing_detection.detection.detection_engine import DetectionEngine
from singing_detection.detection.detection_pipeline import \
    SingingDetectionPipeline
from singing_detection.detection.feature_engineering import FeatureEngineer
from singing_detection.detection.interlude_analysis import InterludeAnalyzer

__all__ = [
    'DetectionEngine',
    'FeatureEngineer',
    'InterludeAnalyzer',
    'SingingDetectionPipeline',
]

