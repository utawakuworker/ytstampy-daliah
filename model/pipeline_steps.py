import json
import os
from typing import Any, Dict

import pandas as pd

from model.data_models import AnalysisResults, Segment
from singing_detection.audio.feature_extraction import FeatureExtractorFacade
from singing_detection.audio.loader import AudioLoaderFactory
from singing_detection.detection.detection_pipeline import \
    SingingDetectionPipeline
from singing_detection.identification.song_identifier import SongIdentifier
from singing_detection.segments.segment_processor import (SegmentFilter,
                                                          SegmentRefiner)


class PipelineStep:
    def run(self, context: Dict[str, Any]) -> bool:
        raise NotImplementedError

class AudioLoaderStep(PipelineStep):
    def __init__(self, config):
        self.config = config

    def run(self, context: Dict[str, Any]) -> bool:
        source = self.config.get('url') or self.config.get('file')
        if not source:
            context['error'] = "No audio source (URL or file) specified."
            return False
        loader = AudioLoaderFactory.create_loader(source, output_dir=self.config.get('output_dir', './output_analysis_model'))
        y, sr = loader.load_audio()
        context['y'] = y
        context['sr'] = sr
        context['audio_path'] = loader.file_path
        context['total_duration'] = loader.duration
        return True

class FeatureExtractionStep(PipelineStep):
    def __init__(self, config):
        self.config = config

    def run(self, context: Dict[str, Any]) -> bool:
        enable_hpss_flag = self.config.get('enable_hpss', True)
        extractor = FeatureExtractorFacade(
            include_pitch=False,
            enable_hpss=enable_hpss_flag,
            window_size_seconds=2.0
        )
        feature_df = extractor.extract_all_features(
            context['y'], context['sr']
        )
        context['feature_df'] = feature_df
        if feature_df.empty:
            context['error'] = "No features extracted."
            return False
        return True

class SegmentDetectionStep(PipelineStep):
    def __init__(self, config):
        self.config = config

    def run(self, context: Dict[str, Any]) -> bool:
        if 'feature_df' not in context or context['feature_df'].empty:
            context['initial_segments'] = []
            context['frame_df_with_states'] = None
            return True
        ref_dur = self.config.get('ref_duration', 2.0)
        sing_start = self.config.get('singing_ref_time', 0)
        nsing_start = self.config.get('non_singing_ref_time', 0)
        total_dur = context.get('total_duration', float('inf'))
        singing_ref = (max(0, sing_start), min(total_dur, sing_start + ref_dur))
        non_singing_ref = (max(0, nsing_start), min(total_dur, nsing_start + ref_dur))
        params = {
            'threshold': self.config.get('hmm_threshold', 0.55),
            'min_duration': self.config.get('min_segment_duration', 10.0),
            'min_gap': self.config.get('min_segment_gap', 1.5),
            'dim_reduction': self.config.get('dim_reduction', 'pca'),
            'n_components': self.config.get('n_components', 4),
            'verbose': False
        }
        segments, results = SingingDetectionPipeline.run(
            context['feature_df'],
            singing_ref, non_singing_ref, params
        )
        context['initial_segments'] = segments
        # Optionally, you can reconstruct a DataFrame similar to frame_df_with_states if needed
        # For now, just store the results dict
        context['frame_df_with_states'] = results
        return True

class SegmentProcessingStep(PipelineStep):
    def __init__(self, config):
        self.config = config

    def run(self, context: Dict[str, Any]) -> bool:
        initial_segments = context.get('initial_segments')
        if initial_segments is None or not initial_segments:
            context['final_segments'] = []
            return True
        segment_refiner = SegmentRefiner()
        segment_filter = SegmentFilter()
        refined_segments = segment_refiner.process_segments(
            segments=initial_segments,
            y=context['y'],
            sr=context['sr']
        )
        filtered_segments = segment_filter.process_segments(
            segments=refined_segments,
            y=context['y'],
            sr=context['sr'],
            min_duration=self.config.get('min_segment_duration', 5.0),
            merge_threshold=self.config.get('merge_threshold', 0.5),
            verbose=False
        )
        final_segment_objects = [Segment(start=s[0], end=s[1]) for s in filtered_segments]
        context['final_segments'] = final_segment_objects
        return True

class SongIdentificationStep(PipelineStep):
    def __init__(self, config):
        self.config = config

    def run(self, context: Dict[str, Any]) -> bool:
        final_segments = context.get('final_segments')
        audio_path = context.get('audio_path')
        if not final_segments:
            context['identification_results'] = []
            return True
        if not audio_path:
            context['error'] = "Cannot identify songs, audio path is missing."
            context['identification_results'] = []
            return False
        api_key = self.config.get('gemini_api_key')
        if not api_key:
            context['identification_results'] = []
            return True
        identifier = SongIdentifier(
            audio_path=audio_path,
            output_dir=self.config.get('output_dir', './output_analysis_model'),
            whisper_model=self.config.get('whisper_model', 'base'),
            gemini_api_key=api_key
        )
        identification_results = identifier.identify_songs(
            segments=final_segments,
            min_segment_duration=self.config.get('min_duration_for_id', 30.0),
            max_segment_duration=self.config.get('max_duration_for_id', 60.0),
            verbose=False
        )
        context['identification_results'] = identification_results
        return True

class VisualizationStep(PipelineStep):
    def __init__(self, config):
        self.config = config

    def run(self, context: Dict[str, Any]) -> bool:
        y = context.get('y')
        sr = context.get('sr')
        processed_segments = context.get('final_segments')
        feature_df = context.get('frame_df_with_states')
        base_filename = context.get('base_filename', 'analysis')
        self.config.get('output_dir', './output_analysis_model')
        if y is None or sr is None or processed_segments is None or feature_df is None:
            return True  # Not an error, just nothing to visualize
        try:
            pass

            from singing_detection.visualization.plots import (
                plot_feature_comparison, plot_waveform_with_segments)

            # Waveform plot
            plot_waveform_with_segments(y, sr, processed_segments, title=f"Detected Segments - {base_filename}")
            # plt.savefig(os.path.join(output_dir, f"{base_filename}_waveform.png"))
            # plt.close()
            # Feature comparison plot
            plot_feature_comparison(feature_df, processed_segments, features=['rms_mean', 'harmonic_ratio_mean', 'singing_probability', 'hmm_state'])
            # plt.savefig(os.path.join(output_dir, f"{base_filename}_features.png"))
            # plt.close()
        except ImportError:
            pass  # Visualization skipped if matplotlib not available
        except Exception:
            pass  # Ignore visualization errors for pipeline robustness
        return True

class SaveResultsStep(PipelineStep):
    def __init__(self, config):
        self.config = config

    def run(self, context: Dict[str, Any]) -> bool:
        base_filename = context.get('base_filename')
        output_dir = self.config.get('output_dir', './output_analysis_model')
        final_segments = context.get('final_segments', [])
        identification_results = context.get('identification_results', [])
        total_duration = context.get('total_duration')
        if not base_filename:
            return True  # Nothing to save
        # Save CSV
        if self.config.get('save_results_dataframe', False):
            try:
                if final_segments:
                    segments_df = pd.DataFrame([
                        {'start': seg.start, 'end': seg.end, 'duration': seg.duration}
                        for seg in final_segments
                    ])
                    csv_path = os.path.join(output_dir, f"{base_filename}_segments.csv")
                    segments_df.to_csv(csv_path, index_label='segment_index', float_format='%.3f')
            except Exception:
                pass
        # Save JSON
        if self.config.get('save_results_json', False):
            try:
                results_obj = AnalysisResults(
                    total_duration=total_duration,
                    final_segments=final_segments,
                    identification_results=identification_results
                )
                import dataclasses
                serializable_data = dataclasses.asdict(results_obj)
                json_path = os.path.join(output_dir, f"{base_filename}_results.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_data, f, indent=4, default=str)
            except Exception:
                pass
        return True 