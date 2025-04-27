# Singing Detection System - SOLID Architecture

## System Architecture

The singing detection system follows SOLID principles with the following major components:

### Audio Processing
- `AudioLoader`: Abstract base class for audio loading
  - `LocalAudioLoader`: For local audio files
  - `YouTubeAudioLoader`: For YouTube URLs
  - `AudioLoaderFactory`: Creates appropriate loader based on source

### Feature Extraction
- `FeatureExtractor`: Abstract base class for feature extraction
  - `SpectralFeatureExtractor`: For spectral features
  - `MFCCFeatureExtractor`: For MFCC features
  - `PitchFeatureExtractor`: For pitch-related features
  - `HarmonicFeatureExtractor`: For harmonic features
  - `FeatureExtractorFacade`: Manages all extractors through single interface

### Detection (Modular Pipeline)
- `FeatureEngineer`: Handles feature scaling, reduction, clustering, and reference info extraction.
- `DetectionEngine`: Stateless class for HMM fit/predict, segment finding, evaluation, and merging.
- `InterludeAnalyzer`: Analyzes and merges interludes between singing segments.
- `SingingDetectionPipeline`: Orchestrates the full detection process using the above modules.

### Segment Processing
- `SegmentProcessor`: Abstract base class for segment processing
  - `SegmentValidator`: Validates detected segments
  - `SegmentFilter`: Filters out invalid segments
  - `SegmentRefiner`: Refines segment boundaries
  - `SegmentMerger`: Merges related segments
  - `SegmentProcessingPipeline`: Manages multiple processors

### Usage Example

```python
from singing_detection.detection import SingingDetectionPipeline

# features_df: DataFrame with extracted features (must include 'time' column)
# singing_ref, non_singing_ref: (start, end) tuples in seconds
# params: dict with detection parameters

segments, results = SingingDetectionPipeline.run(
    features_df,
    singing_ref=(10, 12),
    non_singing_ref=(0, 2),
    params={
        'threshold': 0.6,
        'min_duration': 2.0,
        'min_gap': 1.5,
        'dim_reduction': 'pca',
        'n_components': 4,
        'verbose': True,
        'interlude_threshold': 0.3,
        'max_interlude_duration': 20.0,
    }
)
print(segments)
```

- `segments`: List of (start, end) tuples for detected singing segments.
- `results`: Dictionary with intermediate arrays (times, features, clusters, states, posteriors, evaluated segments).

### Customization
- All main detection parameters are passed via the `params` dictionary.
- You can plug in your own feature extraction or segment post-processing if needed.

### Changelog
- **v2.0.0**: Detection engine is now fully modular. The old monolithic engine (`HMMDetectionEngine`, `ClusterEnhancedHMMDetectionEngine`) has been removed. Use `SingingDetectionPipeline` for all detection tasks.

### Troubleshooting
- Ensure all dependencies are installed: `numpy`, `pandas`, `scikit-learn`, `hmmlearn`, `fastdtw`, etc.
- Input DataFrame must include a `time` column and relevant features.
- For UMAP-based reduction, install `umap-learn`. 