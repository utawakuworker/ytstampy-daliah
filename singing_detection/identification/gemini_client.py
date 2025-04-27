import requests
from typing import Optional, Dict, Any
import json

class GeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    def correct_lyrics(self, transcript: str) -> str:
        if not self.api_key:
            return transcript
        prompt = f"""
        The following is an automatically transcribed song lyric from an audio recognition system. 
        It may contain errors, incorrect words, or grammar issues due to audio quality, background noise, 
        or transcription limitations:

        {transcript}

        Please correct and clean up ONLY the obvious errors in these lyrics. DO NOT completely rewrite them.
        Focus on:
        1. Fixing obvious misspellings
        2. Correcting grammar only where it's clearly wrong
        3. Adjusting punctuation for better readability
        4. Maintaining the original meaning and wording whenever possible

        Return ONLY the corrected lyrics text, with no explanations or metadata.
        """
        response_text = self._call_gemini_api(prompt)
        cleaned_response = response_text.strip()
        if cleaned_response.startswith('"') and cleaned_response.endswith('"'):
            cleaned_response = cleaned_response[1:-1]
        if not cleaned_response:
            return transcript
        return cleaned_response

    def identify_song(self, lyrics: str, start: float, end: float) -> str:
        formatted_start = self._format_time(start)
        formatted_end = self._format_time(end)
        prompt = f"""
        I have a song segment from {formatted_start} to {formatted_end}. Below are the lyrics from this segment:

        {lyrics}

        Please identify this song based on the lyrics, providing the following information:
        1. Song title
        2. Artist/band
        3. Confidence level (high, medium, low)

        Respond in JSON format with fields: title, artist, confidence, and explanation.
        If you cannot identify the song with confidence, provide your best guess and mark the confidence as \"low\". If you are highly uncertain, return null or empty strings for title and artist.
        """
        return self._call_gemini_api(prompt)

    def _call_gemini_api(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
            "tools": [{"googleSearch": {}}],
            "generationConfig": {"temperature": 0, "topP": 0.9, "topK": 40}
        }
        response = requests.post(
            f"{self.api_url}?key={self.api_key}",
            headers=headers,
            json=data
        )
        if response.status_code == 200:
            response_json = response.json()
            try:
                candidate = response_json.get("candidates", [{}])[0]
                content = candidate.get("content", {})
                parts = content.get("parts", [{}])
                full_response_text = ""
                for part in parts:
                    if "text" in part:
                        full_response_text += part["text"] + "\n"
                return full_response_text.strip()
            except (KeyError, IndexError, TypeError) as e:
                print(f"Error parsing Gemini response: {e}")
                return response.text
        else:
            print(f"Gemini API error: {response.status_code} - {response.text}")
            return f"API Error: {response.status_code}"

    def _format_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}" 