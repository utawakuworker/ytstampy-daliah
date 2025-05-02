from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SongIdentificationResult:
    """Represents the result of attempting to identify a single song segment using Gemini."""
    title: Optional[str] = None
    artist: Optional[str] = None
    confidence: str = "low"  # Default to low confidence
    explanation: Optional[str] = None
    refined_lyrics_used: Optional[str] = None # Lyrics actually used for identification
    error: Optional[str] = None # Records any error during identification API call or parsing

@dataclass
class Segment:
    """Represents a detected segment with start and end times."""
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

@dataclass
class SegmentIdentification:
    """Combines a detected segment with its transcription and identification result."""
    segment: Segment # Use the Segment dataclass
    transcript: Optional[str] = None # Raw transcript from Whisper
    identification: Optional[SongIdentificationResult] = None # Result from Gemini ID

@dataclass
class AnalysisResults:
    """Represents the complete results of a singing analysis pipeline run."""
    total_duration: Optional[float] = None
    final_segments: List[Segment] = field(default_factory=list) # List of detected singing segments
    # identification_results contains info for segments processed for ID (might be fewer than final_segments)
    identification_results: List[SegmentIdentification] = field(default_factory=list)
    # Optional: Add raw pipeline output or other metadata if needed
    # raw_output: Optional[Dict[str, Any]] = None 