import json
import os
import glob
import locale
from typing import List # Import List type hint

class LocalizationManager:
    """Manages loading and retrieving translated strings."""

    def __init__(self, locales_dir="locales", default_lang="en"):
        # Try to find the 'locales' directory relative to this file's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir) # Assumes gui/ is one level down from root
        self.locales_dir = os.path.join(project_root, locales_dir)

        self.default_lang = default_lang
        self.translations = {}
        self.current_lang = default_lang # Initialize before load/detect
        self._load_translations()

        # Attempt to set system language, handle fallbacks
        detected_lang = self._detect_system_language()
        if detected_lang in self.translations:
             self.set_language(detected_lang)
        elif self.default_lang in self.translations:
             self.set_language(self.default_lang)
        else: # Fallback if detected and default are unavailable
             first_available = next(iter(self.translations.keys())) if self.translations else 'en'
             print(f"Warning: Detected ('{detected_lang}') and default ('{self.default_lang}') languages not available. Falling back to '{first_available}'.")
             self.set_language(first_available)

    def _load_translations(self):
        """Loads all .json files from the locales directory."""
        self.translations = {}
        if not os.path.isdir(self.locales_dir):
             print(f"Warning: Locales directory not found at '{self.locales_dir}'")
             return

        json_pattern = os.path.join(self.locales_dir, "*.json")
        for file_path in glob.glob(json_pattern):
            lang_code = os.path.splitext(os.path.basename(file_path))[0]
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.translations[lang_code] = json.load(f)
                print(f"Loaded translations for: {lang_code}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load translations from {file_path}: {e}")

    def _detect_system_language(self):
        """Attempts to detect the system language code (e.g., 'en', 'ko')."""
        try:
            system_locale, _ = locale.getdefaultlocale()
            if system_locale:
                lang_code = system_locale.split('_')[0].lower()
                print(f"Detected system language code: {lang_code}")
                return lang_code
        except Exception as e:
            print(f"Could not detect system language: {e}")
        print(f"Could not detect system language.")
        return None # Return None if detection fails

    def set_language(self, lang_code: str):
        """Sets the current language for translations."""
        if lang_code in self.translations:
            self.current_lang = lang_code
            print(f"Language set to: {self.current_lang}")
        else:
            # Avoid changing language if the requested one isn't loaded
            print(f"Warning: Language '{lang_code}' not found or not loaded. Keeping '{self.current_lang}'.")

    # --- Methods for language selection UI ---
    def get_available_languages(self) -> List[str]:
        """Returns a sorted list of loaded language codes."""
        return sorted(list(self.translations.keys()))

    def get_current_language(self) -> str:
        """Returns the currently active language code."""
        # Ensure current_lang is always valid based on loaded translations
        if self.current_lang not in self.translations:
             # Fallback logic if current_lang became invalid somehow
             if self.default_lang in self.translations:
                  self.current_lang = self.default_lang
             elif self.translations:
                  self.current_lang = next(iter(sorted(self.translations.keys())))
             else:
                  self.current_lang = 'en' # Absolute fallback
        return self.current_lang
    # --- End methods for language selection ---

    def tr(self, key: str, **kwargs) -> str:
        """
        Retrieves the translated string for the given key and current language.
        Falls back to default language or the key itself if not found.
        Supports basic keyword argument formatting.
        """
        lang_dict = self.translations.get(self.current_lang)
        default_dict = self.translations.get(self.default_lang)
        translation = key # Default to key if not found anywhere

        if lang_dict and key in lang_dict:
            translation = lang_dict[key]
        elif default_dict and key in default_dict:
            translation = default_dict[key] # Fallback to default language

        try:
            # Attempt basic formatting if keyword args are provided
            if kwargs:
                return translation.format(**kwargs)
            else:
                return translation
        except (KeyError, IndexError, Exception) as e:
             print(f"Error formatting translation for key '{key}' with args {kwargs}: {e}. Returning raw: {translation}")
             return translation # Return unformatted string on error
