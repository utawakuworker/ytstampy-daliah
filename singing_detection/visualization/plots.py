import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import librosa
import librosa.display
import pandas as pd
from matplotlib.patches import Rectangle
from IPython.display import HTML

def plot_waveform_with_segments(y, sr, segments, title="Detected Singing Segments", 
                               reference_segments=None, figsize=(14, 6)):
    """
    Plot audio waveform with highlighted singing segments.
    
    Args:
        y: Audio signal
        sr: Sample rate
        segments: List of (start, end) tuples of detected segments
        title: Plot title
        reference_segments: Optional dictionary of reference segments
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot waveform
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    plt.title(title)
    
    # Color palette for segments
    colors = ['blue', 'green', 'purple', 'orange', 'red', 'cyan', 'magenta']
    
    # Highlight detected segments
    for i, (start, end) in enumerate(segments):
        color = colors[i % len(colors)]
        plt.axvspan(start, end, color=color, alpha=0.2)
        
        # Add time labels
        mid_point = (start + end) / 2
        duration = end - start
        label = f"{start:.1f}-{end:.1f} ({duration:.1f}s)"
        plt.text(mid_point, 0.9, label, 
                horizontalalignment='center', 
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Add reference segments if provided
    if reference_segments:
        for name, (start, end) in reference_segments.items():
            if name == 'singing':
                plt.axvspan(start, end, color='green', alpha=0.3, label=f'{name} ref')
            else:
                plt.axvspan(start, end, color='red', alpha=0.3, label=f'{name} ref')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    if reference_segments:
        plt.legend()
    plt.tight_layout()
    plt.show()

def plot_segmentation_results(results, audio_data=None, sr=None, figsize=(14, 8)):
    """
    Plot comprehensive segmentation results with optional audio waveform.
    
    Args:
        results: Dictionary of segmentation results
        audio_data: Optional audio signal for waveform visualization
        sr: Sample rate for audio
        figsize: Figure size
    """
    # Extract data from results
    segments = results.get('segments', [])
    method_used = results.get('method_used', 'unknown')
    duration = results.get('duration', 0)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Determine number of subplots
    n_plots = 2 if audio_data is not None else 1
    
    # Plot audio waveform if provided
    if audio_data is not None and sr is not None:
        plt.subplot(n_plots, 1, 1)
        librosa.display.waveshow(audio_data, sr=sr, alpha=0.6)
        plt.title(f"Audio Waveform with Detected Segments (Method: {method_used})")
        
        # Highlight detected segments
        colors = ['blue', 'green', 'purple', 'orange', 'red']
        for i, (start, end) in enumerate(segments):
            color = colors[i % len(colors)]
            plt.axvspan(start, end, color=color, alpha=0.2)
    
    # Plot segment timeline
    plt.subplot(n_plots, 1, n_plots)
    
    # Create a colorful timeline visualization
    y_pos = 0
    height = 0.8
    
    # Create colored blocks for singing segments
    for i, (start, end) in enumerate(segments):
        color = plt.cm.viridis(i / max(1, len(segments) - 1))
        plt.gca().add_patch(Rectangle((start, y_pos), end - start, height, 
                                     facecolor=color, alpha=0.8))
        
        # Add segment label
        mid_point = (start + end) / 2
        segment_duration = end - start
        plt.text(mid_point, y_pos + height/2, f"{segment_duration:.1f}s", 
                horizontalalignment='center', verticalalignment='center',
                fontweight='bold', color='white')
    
    # Set axis limits and labels
    plt.xlim(0, duration)
    plt.ylim(0, y_pos + height + 0.2)
    plt.xlabel('Time (seconds)')
    plt.yticks([])
    plt.title(f"Singing Segments Timeline ({len(segments)} segments)")
    
    # Add time markers
    time_markers = np.arange(0, duration, 60)  # Every minute
    if len(time_markers) > 1:
        for t in time_markers:
            plt.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
            m, s = divmod(int(t), 60)
            plt.text(t, y_pos + height + 0.1, f"{m:02d}:{s:02d}", 
                    horizontalalignment='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def plot_feature_comparison(frame_df, segments=None, features=None, figsize=(14, 10)):
    """
    Plot multiple feature comparisons with optional segment highlighting.
    
    Args:
        frame_df: DataFrame with frame features
        segments: Optional list of (start, end) tuples to highlight
        features: List of feature column names to plot
        figsize: Figure size
    """
    if features is None:
        # Default features to plot
        features = [
            'rms_mean', 'contrast_mean', 'harmonic_ratio_mean', 
            'flatness_mean', 'singing_score'
        ]
        # Only include features that exist in the DataFrame
        features = [f for f in features if f in frame_df.columns]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Number of features to plot
    n_features = len(features)
    
    # Plot each feature
    for i, feature in enumerate(features):
        plt.subplot(n_features, 1, i+1)
        plt.plot(frame_df['time'], frame_df[feature], label=feature)
        
        # Highlight segments if provided
        if segments:
            for start, end in segments:
                plt.axvspan(start, end, color='green', alpha=0.2)
        
        plt.ylabel(feature)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # Only show x-label for bottom plot
        if i == n_features - 1:
            plt.xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()

def plot_singing_score_distribution(frame_df, segments=None, score_column='singing_score', figsize=(14, 6)):
    """
    Plot singing score distribution and histogram.
    
    Args:
        frame_df: DataFrame with frame features
        segments: Optional list of (start, end) tuples to analyze separately
        score_column: Column name for the singing score
        figsize: Figure size
    """
    if score_column not in frame_df.columns:
        print(f"Error: '{score_column}' column not found in DataFrame.")
        return
    
    plt.figure(figsize=figsize)
    
    # Time series plot
    plt.subplot(1, 2, 1)
    plt.plot(frame_df['time'], frame_df[score_column], label=score_column)
    
    # Highlight segments if provided
    if segments:
        for start, end in segments:
            plt.axvspan(start, end, color='green', alpha=0.2)
    
    plt.title(f'{score_column} Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel(score_column)
    plt.grid(True, alpha=0.3)
    
    # Histogram
    plt.subplot(1, 2, 2)
    
    # All scores
    plt.hist(frame_df[score_column], bins=30, alpha=0.5, label='All frames')
    
    # Segment scores if provided
    if segments:
        segment_mask = np.zeros(len(frame_df), dtype=bool)
        for start, end in segments:
            segment_mask |= (frame_df['time'] >= start) & (frame_df['time'] <= end)
        
        if np.any(segment_mask):
            plt.hist(frame_df.loc[segment_mask, score_column], bins=30, alpha=0.5, label='Detected segments')
    
    plt.title(f'{score_column} Distribution')
    plt.xlabel(score_column)
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def animate_detection_process(frame_df, threshold=0.6, figsize=(14, 7), interval=50):
    """
    Create an animation showing the singing detection process frame by frame.
    
    Args:
        frame_df: DataFrame with frame features
        threshold: Detection threshold
        figsize: Figure size
        interval: Animation interval in milliseconds
        
    Returns:
        HTML: Animation for display in notebook
    """
    # Get data for animation
    times = frame_df['time'].values
    
    # Use singing_score if available, otherwise use harmonic_ratio
    if 'singing_score' in frame_df:
        scores = frame_df['singing_score'].values
        score_name = 'Singing Score'
    elif 'harmonic_ratio_mean' in frame_df:
        scores = frame_df['harmonic_ratio_mean'].values
        score_name = 'Harmonic Ratio'
    else:
        print("No suitable score column found for animation.")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Initialize plot
    line, = ax.plot([], [], lw=2)
    threshold_line = ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.7)
    current_point, = ax.plot([], [], 'ro', markersize=8)
    
    # Detection status text
    detection_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                           bbox=dict(facecolor='white', alpha=0.8))
    
    # Configure axes
    ax.set_xlim(0, max(times))
    ax.set_ylim(0, max(1.1, max(scores) * 1.1))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(score_name)
    ax.set_title(f'Singing Detection Process (Threshold = {threshold:.2f})')
    ax.grid(True, alpha=0.3)
    
    # Animation function
    def animate(i):
        current_time = times[:i+1]
        current_scores = scores[:i+1]
        
        line.set_data(current_time, current_scores)
        
        if i < len(times):
            current_point.set_data(times[i], scores[i])
            
            # Update detection status
            if scores[i] > threshold:
                status = 'SINGING DETECTED'
                color = 'green'
            else:
                status = 'NO SINGING'
                color = 'red'
                
            detection_text.set_text(status)
            detection_text.set_color(color)
        
        return line, current_point, detection_text
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(times), 
                                 interval=interval, blit=True)
    
    plt.close()  # Prevent duplicate display
    return HTML(anim.to_jshtml()) 