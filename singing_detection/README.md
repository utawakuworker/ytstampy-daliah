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

### Detection
- `DetectionEngine`: Abstract base class for singing detection
  - `HMMDetectionEngine`: Basic HMM-based detection
  - `ClusterEnhancedHMMDetectionEngine`: Detection with clustering enhancement

### Segment Processing
- `SegmentProcessor`: Abstract base class for segment processing
  - `SegmentValidator`: Validates detected segments
  - `SegmentFilter`: Filters out invalid segments
  - `SegmentRefiner`: Refines segment boundaries
  - `SegmentMerger`: Merges related segments
  - `SegmentProcessingPipeline`: Manages multiple processors

### Usage

See `test.py` for a complete example of the new architecture. 