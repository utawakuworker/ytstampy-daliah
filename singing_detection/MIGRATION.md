# Migration Guide: Moving to SOLID Architecture

This guide helps you migrate from the legacy function-based API to the new SOLID class-based architecture.

## Key Changes

1. **Audio Processing**
   - Old: `extract_audio_features_sequential()` function
   - New: `FeatureExtractorFacade` class

   ```python
   # Old way
   frame_df = extract_audio_features_sequential(y, sr, include_pitch=True)
   
   # New way
   extractor = FeatureExtractorFacade(include_pitch=True)
   frame_df = extractor.extract_all_features(y, sr)
   ```

2. **Audio Loading**
   - Old: `librosa.load()` and `download_youtube_audio()` functions
   - New: `AudioLoaderFactory` and concrete loader classes

   ```python
   # Old way
   if 'youtube' in source:
       audio_path = download_youtube_audio(source)
       y, sr = librosa.load(audio_path)
   else:
       y, sr = librosa.load(source)
   
   # New way
   loader = AudioLoaderFactory.create_loader(source)
   y, sr = loader.load_audio()
   ```

3. **Detection**
   - Old: `detect_singing_with_cluster_enhanced_hmm()` function
   - New: `ClusterEnhancedHMMDetectionEngine` class

   ```python
   # Old way
   segments, frame_df = detect_singing_with_cluster_enhanced_hmm(
       y, sr, frame_df, singing_segment, non_singing_segment, params
   )
   
   # New way
   detector = ClusterEnhancedHMMDetectionEngine()
   segments, frame_df = detector.detect_singing(
       y, sr, frame_df, singing_segment, non_singing_segment, **params
   )
   ```

4. **Segment Processing**
   - Old: Individual processing functions
   - New: `SegmentProcessingPipeline` with processor classes

   ```python
   # Old way
   segments = post_validate_segments(segments, y, sr, ...)
   segments = filter_short_segments(segments, ...)
   
   # New way
   pipeline = SegmentProcessingPipeline()
   pipeline.add_processor(SegmentValidator())
   pipeline.add_processor(SegmentFilter())
   segments = pipeline.process(segments, y, sr, **params)
   ```

See `test.py` for a complete example of using the new architecture. 