import numpy as np
import librosa
import matplotlib.pyplot as plt
from IPython.display import display, Audio, HTML

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def play_audio_segment(y, sr, start_sec, end_sec, display_title="Audio Segment"):
    """Play an audio segment within a Jupyter notebook."""
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    segment = y[start_sample:end_sample]
    display(HTML(f"<h4>{display_title}</h4>"))
    display(Audio(segment, rate=sr))

def export_results_as_video_chapters(results, output_path='chapters.txt'):
    """Export results in a format suitable for video chapter markers."""
    with open(output_path, 'w') as f:
        for item in results:
            # Ensure keys exist before accessing
            start_time = item.get('start_time', '00:00:00')
            song_title = item.get('song_title', 'Unknown Song')
            artist = item.get('artist', 'Unknown Artist')
            f.write(f"{start_time} - {song_title} by {artist}\n")
    return output_path 