# Voice Processing Module

## Overview
Complete voice pipeline with NVIDIA Riva.

## Files

### `wake_word.py`
Always-on wake word detection.
- Detects "Hey Robot", "OK Robot"
- Low-power continuous listening

### `asr.py`
Multi-language speech recognition.
- Languages: en-US, es-ES, de-DE, fr-FR, zh-CN, ja-JP
- Streaming and batch modes

### `tts.py`
Emotional text-to-speech.
- Emotions: neutral, happy, sad, excited, concerned
- Configurable voice, pitch, speed

### `noise_filter.py`
Industrial noise suppression.
- Adaptive filtering
- Factory/warehouse profiles

## Pipeline Flow

```
Mic → NoiseFilter → WakeWord → ASR → Intent → Response → TTS → Speaker
```

## Usage

```python
from voice import MultiLanguageASR, EmotionalTTS

asr = MultiLanguageASR(riva_url="localhost:50051")
result = asr.transcribe(audio_bytes, language="en-US")

tts = EmotionalTTS()
audio = tts.synthesize("Task completed!", emotion="happy")
```
