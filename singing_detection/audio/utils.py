#interestingly, no imports are needed for this file

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


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