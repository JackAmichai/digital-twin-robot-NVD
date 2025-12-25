#!/usr/bin/env python3
"""
Multi-Language ASR Client with Auto-Detection

This module extends the ASR client to support multiple languages with
automatic language detection. NVIDIA Riva supports 7+ languages including:
- English (en-US, en-GB)
- Spanish (es-ES, es-US)
- German (de-DE)
- French (fr-FR)
- Italian (it-IT)
- Portuguese (pt-BR)
- Russian (ru-RU)
- Mandarin Chinese (zh-CN)
- Japanese (ja-JP)
- Korean (ko-KR)
- Hindi (hi-IN)

Features:
- Automatic language detection from audio
- Language-specific command translation to English
- Per-user language preference tracking
- Fallback to English if detection fails

Example Usage:
    client = MultiLanguageASR()
    
    # Auto-detect language
    text, lang = client.transcribe_with_detection(audio_data)
    print(f"Detected {lang}: {text}")
    
    # Force specific language
    text = client.transcribe(audio_data, language="es-ES")
"""

import grpc
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import logging
import os

# Riva imports
try:
    import riva.client
    from riva.client import ASRService
    RIVA_AVAILABLE = True
except ImportError:
    RIVA_AVAILABLE = False


class SupportedLanguage(Enum):
    """
    Supported languages with Riva language codes
    
    Each language has:
    - Riva code for ASR
    - Display name
    - Wake word variants
    - Common command translations
    """
    ENGLISH_US = "en-US"
    ENGLISH_UK = "en-GB"
    SPANISH_ES = "es-ES"
    SPANISH_US = "es-US"
    GERMAN = "de-DE"
    FRENCH = "fr-FR"
    ITALIAN = "it-IT"
    PORTUGUESE = "pt-BR"
    RUSSIAN = "ru-RU"
    MANDARIN = "zh-CN"
    JAPANESE = "ja-JP"
    KOREAN = "ko-KR"
    HINDI = "hi-IN"


@dataclass
class LanguageConfig:
    """Configuration for a specific language"""
    code: str
    display_name: str
    wake_phrases: List[str]
    # Common command translations (native → English)
    command_translations: Dict[str, str] = field(default_factory=dict)


# Language configurations with localized wake phrases and command mappings
LANGUAGE_CONFIGS: Dict[str, LanguageConfig] = {
    "en-US": LanguageConfig(
        code="en-US",
        display_name="English (US)",
        wake_phrases=["hey robot", "ok robot", "hello robot"],
        command_translations={}  # English is our base language
    ),
    "es-ES": LanguageConfig(
        code="es-ES",
        display_name="Español",
        wake_phrases=["oye robot", "hola robot", "robot"],
        command_translations={
            "ir a": "go to",
            "navegar a": "navigate to",
            "ve a": "go to",
            "detente": "stop",
            "para": "stop",
            "avanza": "move forward",
            "retrocede": "move backward",
            "gira a la izquierda": "turn left",
            "gira a la derecha": "turn right",
            "zona de carga": "loading zone",
            "estación de carga": "charging station",
            "almacén": "warehouse",
        }
    ),
    "de-DE": LanguageConfig(
        code="de-DE",
        display_name="Deutsch",
        wake_phrases=["hey roboter", "hallo roboter", "roboter"],
        command_translations={
            "gehe zu": "go to",
            "navigiere zu": "navigate to",
            "fahr zu": "go to",
            "stopp": "stop",
            "halt": "stop",
            "vorwärts": "move forward",
            "rückwärts": "move backward",
            "links abbiegen": "turn left",
            "rechts abbiegen": "turn right",
            "ladestation": "charging station",
            "lagerbereich": "warehouse zone",
        }
    ),
    "fr-FR": LanguageConfig(
        code="fr-FR",
        display_name="Français",
        wake_phrases=["hé robot", "bonjour robot", "robot"],
        command_translations={
            "aller à": "go to",
            "va à": "go to",
            "naviguer vers": "navigate to",
            "arrête": "stop",
            "stop": "stop",
            "avance": "move forward",
            "recule": "move backward",
            "tourne à gauche": "turn left",
            "tourne à droite": "turn right",
            "station de charge": "charging station",
            "zone de stockage": "warehouse zone",
        }
    ),
    "it-IT": LanguageConfig(
        code="it-IT",
        display_name="Italiano",
        wake_phrases=["ehi robot", "ciao robot", "robot"],
        command_translations={
            "vai a": "go to",
            "naviga verso": "navigate to",
            "fermati": "stop",
            "stop": "stop",
            "avanti": "move forward",
            "indietro": "move backward",
            "gira a sinistra": "turn left",
            "gira a destra": "turn right",
            "stazione di ricarica": "charging station",
        }
    ),
    "pt-BR": LanguageConfig(
        code="pt-BR",
        display_name="Português",
        wake_phrases=["ei robô", "olá robô", "robô"],
        command_translations={
            "ir para": "go to",
            "vá para": "go to",
            "navegue para": "navigate to",
            "pare": "stop",
            "parar": "stop",
            "avance": "move forward",
            "recue": "move backward",
            "vire à esquerda": "turn left",
            "vire à direita": "turn right",
            "estação de carregamento": "charging station",
        }
    ),
    "zh-CN": LanguageConfig(
        code="zh-CN",
        display_name="中文",
        wake_phrases=["嘿机器人", "你好机器人", "机器人"],
        command_translations={
            "去": "go to",
            "导航到": "navigate to",
            "停止": "stop",
            "停": "stop",
            "前进": "move forward",
            "后退": "move backward",
            "左转": "turn left",
            "右转": "turn right",
            "充电站": "charging station",
            "仓库": "warehouse",
        }
    ),
    "ja-JP": LanguageConfig(
        code="ja-JP",
        display_name="日本語",
        wake_phrases=["ヘイロボット", "こんにちはロボット", "ロボット"],
        command_translations={
            "行って": "go to",
            "移動して": "navigate to",
            "止まって": "stop",
            "停止": "stop",
            "前進": "move forward",
            "後退": "move backward",
            "左に曲がって": "turn left",
            "右に曲がって": "turn right",
            "充電ステーション": "charging station",
        }
    ),
    "ru-RU": LanguageConfig(
        code="ru-RU",
        display_name="Русский",
        wake_phrases=["эй робот", "привет робот", "робот"],
        command_translations={
            "иди к": "go to",
            "езжай к": "go to",
            "навигация к": "navigate to",
            "стоп": "stop",
            "остановись": "stop",
            "вперёд": "move forward",
            "назад": "move backward",
            "поверни налево": "turn left",
            "поверни направо": "turn right",
            "зарядная станция": "charging station",
        }
    ),
    "ko-KR": LanguageConfig(
        code="ko-KR",
        display_name="한국어",
        wake_phrases=["헤이 로봇", "안녕 로봇", "로봇"],
        command_translations={
            "가": "go to",
            "이동해": "navigate to",
            "멈춰": "stop",
            "정지": "stop",
            "앞으로": "move forward",
            "뒤로": "move backward",
            "왼쪽으로": "turn left",
            "오른쪽으로": "turn right",
            "충전소": "charging station",
        }
    ),
    "hi-IN": LanguageConfig(
        code="hi-IN",
        display_name="हिन्दी",
        wake_phrases=["हे रोबोट", "हैलो रोबोट", "रोबोट"],
        command_translations={
            "जाओ": "go to",
            "जाना": "go to",
            "रुको": "stop",
            "रुकें": "stop",
            "आगे": "move forward",
            "पीछे": "move backward",
            "बाएं मुड़ो": "turn left",
            "दाएं मुड़ो": "turn right",
            "चार्जिंग स्टेशन": "charging station",
        }
    ),
}


@dataclass
class LanguageDetectionResult:
    """Result of language detection"""
    detected_language: str
    confidence: float
    alternatives: List[Tuple[str, float]]  # (language, confidence) pairs


class MultiLanguageASR:
    """
    Multi-language ASR client with automatic language detection
    
    Features:
    - Detects language from audio automatically
    - Supports 11+ languages via NVIDIA Riva
    - Translates commands to English for processing
    - Caches user language preferences
    
    Example:
        asr = MultiLanguageASR(riva_server="localhost:50051")
        
        # Auto-detect language
        result = asr.transcribe_auto(audio_data)
        print(f"Language: {result.language}")
        print(f"Text: {result.transcript}")
        print(f"English: {result.english_translation}")
        
        # Force specific language
        result = asr.transcribe(audio_data, language="es-ES")
    """
    
    def __init__(self, riva_server: str = "localhost:50051",
                 default_language: str = "en-US",
                 enable_translation: bool = True):
        """
        Initialize multi-language ASR client
        
        Args:
            riva_server: Riva gRPC server address
            default_language: Fallback language if detection fails
            enable_translation: Whether to translate to English
        """
        self.riva_server = riva_server
        self.default_language = default_language
        self.enable_translation = enable_translation
        self.logger = logging.getLogger("MultiLanguageASR")
        
        # User language preferences (user_id → language_code)
        self._user_languages: Dict[str, str] = {}
        
        # Language detection models
        self._detection_enabled = True
        
        # Initialize Riva
        self._riva_auth = None
        self._asr_service = None
        
        if RIVA_AVAILABLE:
            self._init_riva()
    
    def _init_riva(self):
        """Initialize Riva client"""
        try:
            self._riva_auth = riva.client.Auth(
                uri=self.riva_server,
                use_ssl=False
            )
            self._asr_service = ASRService(self._riva_auth)
            self.logger.info(f"Multi-language ASR connected to {self.riva_server}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Riva: {e}")
    
    def get_supported_languages(self) -> List[Dict]:
        """
        Get list of supported languages
        
        Returns:
            List of dicts with language info:
            [{"code": "en-US", "name": "English (US)", "wake_phrases": [...]}]
        """
        return [
            {
                "code": config.code,
                "name": config.display_name,
                "wake_phrases": config.wake_phrases
            }
            for config in LANGUAGE_CONFIGS.values()
        ]
    
    def detect_language(self, audio_data: bytes, 
                       sample_rate: int = 16000) -> LanguageDetectionResult:
        """
        Detect language from audio sample
        
        Uses parallel ASR with multiple language models and selects
        the one with highest confidence.
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            sample_rate: Audio sample rate
            
        Returns:
            LanguageDetectionResult with detected language and confidence
        """
        if not self._asr_service:
            return LanguageDetectionResult(
                detected_language=self.default_language,
                confidence=0.0,
                alternatives=[]
            )
        
        # Languages to try for detection
        # Using a subset for efficiency
        detection_languages = ["en-US", "es-ES", "de-DE", "fr-FR", "zh-CN", "ja-JP"]
        
        results: List[Tuple[str, float, str]] = []  # (language, confidence, transcript)
        
        for lang_code in detection_languages:
            try:
                config = riva.client.RecognitionConfig(
                    language_code=lang_code,
                    max_alternatives=1,
                    enable_automatic_punctuation=False,
                    sample_rate_hertz=sample_rate,
                    audio_channel_count=1,
                )
                
                response = self._asr_service.offline_recognize(
                    audio_data,
                    config
                )
                
                if response.results:
                    alt = response.results[0].alternatives[0]
                    confidence = alt.confidence
                    transcript = alt.transcript
                    
                    # Boost confidence if transcript contains known words for this language
                    if self._contains_language_markers(transcript, lang_code):
                        confidence *= 1.2
                    
                    results.append((lang_code, min(confidence, 1.0), transcript))
                    
            except Exception as e:
                self.logger.debug(f"Detection error for {lang_code}: {e}")
        
        # Sort by confidence
        results.sort(key=lambda x: x[1], reverse=True)
        
        if results:
            best_lang, best_conf, _ = results[0]
            alternatives = [(lang, conf) for lang, conf, _ in results[1:4]]
            
            self.logger.info(f"Detected language: {best_lang} ({best_conf:.2f})")
            
            return LanguageDetectionResult(
                detected_language=best_lang,
                confidence=best_conf,
                alternatives=alternatives
            )
        
        return LanguageDetectionResult(
            detected_language=self.default_language,
            confidence=0.0,
            alternatives=[]
        )
    
    def _contains_language_markers(self, text: str, language: str) -> bool:
        """Check if text contains markers specific to a language"""
        text_lower = text.lower()
        
        # Language-specific common words
        markers = {
            "en-US": ["the", "is", "to", "and", "robot"],
            "es-ES": ["el", "la", "de", "que", "robot"],
            "de-DE": ["der", "die", "das", "und", "roboter"],
            "fr-FR": ["le", "la", "de", "et", "robot"],
            "zh-CN": ["的", "是", "在", "机器人"],
            "ja-JP": ["の", "は", "を", "ロボット"],
        }
        
        if language in markers:
            return any(marker in text_lower for marker in markers[language])
        return False
    
    def transcribe(self, audio_data: bytes, 
                   language: str = None,
                   sample_rate: int = 16000) -> Dict:
        """
        Transcribe audio in a specific language
        
        Args:
            audio_data: Raw audio bytes
            language: Language code (e.g., "es-ES"). Uses default if None
            sample_rate: Audio sample rate
            
        Returns:
            Dict with transcript and metadata:
            {
                "transcript": "original text",
                "language": "es-ES",
                "confidence": 0.95,
                "english_translation": "translated text"
            }
        """
        language = language or self.default_language
        
        if not self._asr_service:
            return {
                "transcript": "",
                "language": language,
                "confidence": 0.0,
                "english_translation": ""
            }
        
        try:
            # Get language config for wake phrase boosting
            lang_config = LANGUAGE_CONFIGS.get(language)
            boost_phrases = lang_config.wake_phrases if lang_config else []
            
            config = riva.client.RecognitionConfig(
                language_code=language,
                max_alternatives=1,
                enable_automatic_punctuation=True,
                sample_rate_hertz=sample_rate,
                audio_channel_count=1,
                speech_contexts=[
                    riva.client.SpeechContext(
                        phrases=boost_phrases,
                        boost=10.0
                    )
                ] if boost_phrases else []
            )
            
            response = self._asr_service.offline_recognize(audio_data, config)
            
            if response.results:
                alt = response.results[0].alternatives[0]
                transcript = alt.transcript
                confidence = alt.confidence
                
                # Translate to English if needed
                english_text = transcript
                if self.enable_translation and language != "en-US":
                    english_text = self.translate_to_english(transcript, language)
                
                return {
                    "transcript": transcript,
                    "language": language,
                    "confidence": confidence,
                    "english_translation": english_text
                }
            
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
        
        return {
            "transcript": "",
            "language": language,
            "confidence": 0.0,
            "english_translation": ""
        }
    
    def transcribe_auto(self, audio_data: bytes,
                       sample_rate: int = 16000,
                       user_id: str = None) -> Dict:
        """
        Transcribe audio with automatic language detection
        
        If user_id is provided and has a saved preference, uses that language.
        Otherwise, detects language from audio.
        
        Args:
            audio_data: Raw audio bytes
            sample_rate: Audio sample rate
            user_id: Optional user ID for preference lookup
            
        Returns:
            Dict with transcript, detected language, and translation
        """
        # Check for user preference
        if user_id and user_id in self._user_languages:
            preferred_lang = self._user_languages[user_id]
            self.logger.info(f"Using saved preference: {preferred_lang}")
            return self.transcribe(audio_data, preferred_lang, sample_rate)
        
        # Detect language
        detection = self.detect_language(audio_data, sample_rate)
        
        # Transcribe in detected language
        result = self.transcribe(audio_data, detection.detected_language, sample_rate)
        result["detection_confidence"] = detection.confidence
        result["language_alternatives"] = detection.alternatives
        
        # Save user preference if confident
        if user_id and detection.confidence > 0.8:
            self._user_languages[user_id] = detection.detected_language
            self.logger.info(f"Saved language preference for {user_id}: {detection.detected_language}")
        
        return result
    
    def translate_to_english(self, text: str, source_language: str) -> str:
        """
        Translate command text to English
        
        Uses predefined command mappings for robot commands.
        For general text, returns original (or could use NMT model).
        
        Args:
            text: Text in source language
            source_language: Source language code
            
        Returns:
            English translation or original text
        """
        if source_language == "en-US":
            return text
        
        config = LANGUAGE_CONFIGS.get(source_language)
        if not config:
            return text
        
        text_lower = text.lower()
        result = text_lower
        
        # Apply command translations
        for native, english in config.command_translations.items():
            if native in text_lower:
                result = result.replace(native, english)
        
        return result
    
    def set_user_language(self, user_id: str, language: str):
        """
        Set preferred language for a user
        
        Args:
            user_id: User identifier
            language: Language code (e.g., "es-ES")
        """
        if language in LANGUAGE_CONFIGS:
            self._user_languages[user_id] = language
            self.logger.info(f"Set language for {user_id}: {language}")
        else:
            self.logger.warning(f"Unknown language: {language}")
    
    def get_user_language(self, user_id: str) -> str:
        """Get preferred language for a user"""
        return self._user_languages.get(user_id, self.default_language)
    
    def get_wake_phrases(self, language: str = None) -> List[str]:
        """
        Get wake phrases for a language
        
        Args:
            language: Language code. If None, returns all languages
            
        Returns:
            List of wake phrases
        """
        if language:
            config = LANGUAGE_CONFIGS.get(language)
            return config.wake_phrases if config else []
        
        # Return all wake phrases from all languages
        all_phrases = []
        for config in LANGUAGE_CONFIGS.values():
            all_phrases.extend(config.wake_phrases)
        return all_phrases


# ============================================
# Integration with existing ASR client
# ============================================

def create_multilingual_asr_config(language: str = "en-US") -> dict:
    """
    Create ASR configuration for use with existing asr_client.py
    
    Example:
        from multilingual_asr import create_multilingual_asr_config
        config = create_multilingual_asr_config("es-ES")
        # Use config in existing ASR client
    """
    lang_config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["en-US"])
    
    return {
        "language_code": lang_config.code,
        "wake_phrases": lang_config.wake_phrases,
        "command_translations": lang_config.command_translations,
        "display_name": lang_config.display_name
    }


def main():
    """Demo of multi-language ASR"""
    logging.basicConfig(level=logging.INFO)
    
    print("Multi-Language ASR Demo")
    print("=" * 50)
    
    asr = MultiLanguageASR()
    
    print("\nSupported Languages:")
    for lang in asr.get_supported_languages():
        print(f"  {lang['code']}: {lang['name']}")
        print(f"    Wake phrases: {', '.join(lang['wake_phrases'])}")
    
    print("\nCommand Translations (Spanish → English):")
    es_config = LANGUAGE_CONFIGS["es-ES"]
    for native, english in list(es_config.command_translations.items())[:5]:
        print(f"  '{native}' → '{english}'")
    
    print("\n" + "=" * 50)
    print("To test with audio, provide audio data to:")
    print("  asr.transcribe_auto(audio_bytes)")


if __name__ == "__main__":
    main()
