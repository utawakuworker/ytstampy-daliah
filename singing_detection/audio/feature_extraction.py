from abc import ABC, abstractmethod
from typing import List

import librosa
import numpy as np
import pandas as pd


class FeatureExtractor(ABC):
    """Abstract base class for audio feature extractors."""
    
    def __init__(self, frame_length=2048, hop_length=512):
        """Initialize with frame and hop lengths"""
        self.frame_length = frame_length
        self.hop_length = hop_length
    
    @abstractmethod
    def extract_features(self, y: np.ndarray, sr: int) -> pd.DataFrame:
        """Extract features from audio signal and return as DataFrame."""
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names this extractor provides."""


class SpectralFeatureExtractor(FeatureExtractor):
    """Extractor for spectral features like contrast, flatness, etc."""
    
    def extract_features(self, y: np.ndarray, sr: int) -> pd.DataFrame:
        """Extract spectral features from audio signal."""
        print(f"Spectral feature extraction with frame_length={self.frame_length}, hop_length={self.hop_length}")
        
        # Extract spectral features with consistent frame and hop lengths
        S = np.abs(librosa.stft(y, n_fft=self.frame_length, hop_length=self.hop_length))
        
        # Calculate features directly from the signal for very large frame sizes
        rms = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        
        # Create DataFrame with consistent length
        spectral_df = pd.DataFrame({
            'rms_mean': rms,
            'zcr_mean': zcr,
            'spectral_centroid_mean': spectral_centroid,
            'spectral_bandwidth_mean': spectral_bandwidth,
            'spectral_rolloff_mean': spectral_rolloff,
        })
        
        return spectral_df
    
    @property
    def feature_names(self) -> List[str]:
        return ['rms_mean', 'zcr_mean', 'spectral_centroid_mean', 
                'spectral_bandwidth_mean', 'spectral_rolloff_mean']


class MFCCFeatureExtractor(FeatureExtractor):
    """Extractor for MFCC features."""
    
    def __init__(self, frame_length=2048, hop_length=512, n_mfcc=13):
        super().__init__(frame_length, hop_length)
        self.n_mfcc = n_mfcc
    
    def extract_features(self, y: np.ndarray, sr: int) -> pd.DataFrame:
        """Extract MFCC features from audio signal."""
        print(f"MFCC feature extraction with frame_length={self.frame_length}, hop_length={self.hop_length}")
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, 
            n_mfcc=self.n_mfcc, 
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        
        # Calculate mean and std of MFCCs per frame
        mfcc_mean = np.mean(mfccs, axis=0)
        mfcc_std = np.std(mfccs, axis=0)
        
        # Create DataFrame
        mfcc_df = pd.DataFrame({
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std,
        })
        
        return mfcc_df
    
    @property
    def feature_names(self) -> List[str]:
        return ['mfcc_mean', 'mfcc_std']


class HarmonicFeatureExtractor(FeatureExtractor):
    """Extractor for harmonic features."""
    
    def extract_features(self, y: np.ndarray, sr: int) -> pd.DataFrame:
        """Extract harmonic features from audio signal."""
        print(f"Harmonic feature extraction with frame_length={self.frame_length}, hop_length={self.hop_length}")
        
        # Harmonic-percussive source separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Compute harmonic ratio as the ratio of harmonic energy to total energy
        S_harmonic = np.abs(librosa.stft(y_harmonic, n_fft=self.frame_length, hop_length=self.hop_length))
        S_percussive = np.abs(librosa.stft(y_percussive, n_fft=self.frame_length, hop_length=self.hop_length))
        
        # Calculate energy per frame
        harmonic_energy = np.sum(S_harmonic**2, axis=0)
        percussive_energy = np.sum(S_percussive**2, axis=0)
        total_energy = harmonic_energy + percussive_energy
        
        # Calculate harmonic ratio (avoid division by zero)
        harmonic_ratio = np.zeros_like(total_energy)
        mask = total_energy > 0
        harmonic_ratio[mask] = harmonic_energy[mask] / total_energy[mask]
        
        # Create DataFrame
        harmonic_df = pd.DataFrame({
            'harmonic_ratio_mean': harmonic_ratio,
        })
        
        return harmonic_df
    
    @property
    def feature_names(self) -> List[str]:
        return ['harmonic_ratio_mean']


class PitchFeatureExtractor(FeatureExtractor):
    """Extractor for pitch-related features."""
    
    def __init__(self, frame_length=2048, hop_length=512, fmin=80, fmax=1000):
        super().__init__(frame_length, hop_length)
        self.fmin = fmin
        self.fmax = fmax
    
    def extract_features(self, y: np.ndarray, sr: int) -> pd.DataFrame:
        """Extract pitch features from audio signal."""
        print(f"Pitch feature extraction with frame_length={self.frame_length}, hop_length={self.hop_length}")
        
        # Extract pitch with consistent hop length
        pitches, magnitudes = librosa.piptrack(
            y=y, sr=sr, 
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            fmin=self.fmin, fmax=self.fmax
        )
        
        # Calculate pitch statistics per frame
        pitch_means = []
        pitch_stabilities = []
        
        for t in range(pitches.shape[1]):
            pitches_t = pitches[:, t]
            mags_t = magnitudes[:, t]
            
            # Get top pitches by magnitude
            idx = np.argmax(mags_t)
            pitch_t = pitches_t[idx] if mags_t[idx] > 0 else 0
            
            pitch_means.append(pitch_t)
            
            # Calculate pitch stability (inverse of variance)
            if np.sum(mags_t) > 0:
                # Weight pitches by magnitudes
                weighted_pitches = pitches_t * mags_t
                # Consider only non-zero pitches
                valid_idx = weighted_pitches > 0
                if np.any(valid_idx):
                    pitch_var = np.var(weighted_pitches[valid_idx])
                    stability = 1.0 / (1.0 + pitch_var) if pitch_var > 0 else 1.0
                else:
                    stability = 0.0
            else:
                stability = 0.0
                
            pitch_stabilities.append(stability)
        
        # Create DataFrame
        pitch_df = pd.DataFrame({
            'pitch_mean': pitch_means,
            'pitch_stability_mean': pitch_stabilities,
        })
        
        return pitch_df
    
    @property
    def feature_names(self) -> List[str]:
        return ['pitch_mean', 'pitch_stability_mean']


class FeatureExtractorFacade:
    """
    Facade for extracting various audio features, providing a simplified interface.
    Delegates to specialized feature extractors.
    """
    
    def __init__(self, 
                 include_pitch=True, 
                 enable_hpss=True,
                 frame_length=None, 
                 hop_length=None, 
                 window_size_seconds=1.0):
        """
        Initialize the feature extractor facade.
        
        Args:
            include_pitch: Whether to include pitch-related features
            enable_hpss: Whether to include harmonic features (requires HPSS)
            frame_length: Frame length for feature extraction (if None, calculated from window_size_seconds)
            hop_length: Hop length for feature extraction (if None, calculated as half the frame_length)
            window_size_seconds: Window size in seconds (default: 1.0)
        """
        self.include_pitch = include_pitch
        self.enable_hpss = enable_hpss
        self.window_size_seconds = window_size_seconds
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.feature_rate = None
        
        # Extractors will be initialized in extract_all_features when we know sr
        self.extractors = []
    
    def extract_all_features(self, y, sr):
        """
        Extract all relevant features and combine into a single DataFrame.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            pd.DataFrame: DataFrame with all extracted features
        """
        # Calculate frame and hop lengths based on window size if not provided
        if self.frame_length is None:
            self.frame_length = int(sr * self.window_size_seconds)
            print(f"Setting frame_length to {self.frame_length} samples ({self.window_size_seconds} seconds)")
        
        if self.hop_length is None:
            # Use 50% overlap between windows
            self.hop_length = self.frame_length // 2
            print(f"Setting hop_length to {self.hop_length} samples (50% overlap)")
        
        # Initialize the specialized extractors with the calculated parameters
        self.extractors = [
            SpectralFeatureExtractor(self.frame_length, self.hop_length),
            MFCCFeatureExtractor(self.frame_length, self.hop_length),
        ]
        
        # Conditionally add HarmonicFeatureExtractor
        if self.enable_hpss:
            print("HPSS enabled, adding HarmonicFeatureExtractor.")
            self.extractors.append(HarmonicFeatureExtractor(self.frame_length, self.hop_length))
        else:
            print("HPSS disabled, skipping HarmonicFeatureExtractor.")
        
        if self.include_pitch:
            self.extractors.append(PitchFeatureExtractor(self.frame_length, self.hop_length))
        
        # Calculate time points for frames
        frame_time = librosa.frames_to_time(
            np.arange(1 + len(y) // self.hop_length), 
            sr=sr, 
            hop_length=self.hop_length
        )
        
        # Store the feature rate for later use
        self.feature_rate = sr / self.hop_length
        
        print(f"Feature extraction: {len(frame_time)} frames at {self.feature_rate:.1f} Hz")
        
        # Initialize DataFrame with time column
        df = pd.DataFrame({'time': frame_time})
        
        # Extract features from each extractor and combine
        dfs_to_combine = [df]
        
        for extractor in self.extractors:
            print(f"Using {extractor.__class__.__name__}...")
            feature_df = extractor.extract_features(y, sr)
            
            # Only combine if lengths match
            if len(feature_df) == len(df):
                dfs_to_combine.append(feature_df)
            else:
                print(f"Warning: {extractor.__class__.__name__} produced {len(feature_df)} frames, expected {len(df)}")
        
        # Combine all valid DataFrames
        result_df = pd.concat(dfs_to_combine, axis=1)
        
        # Remove any duplicate columns (like 'time' if it appears in multiple dataframes)
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        
        return result_df
    
    def get_feature_rate(self):
        """
        Get the feature frame rate (frames per second).
        
        Returns:
            float: Feature frame rate
        """
        if self.feature_rate is None:
            raise ValueError("Feature rate not available. Run extract_all_features first.")
        return self.feature_rate 