#!/usr/bin/env python3
"""
Ambient Noise Filtering for Warehouse Environments

Provides sophisticated audio preprocessing to improve ASR accuracy in noisy
industrial settings. Implements multiple filtering techniques optimized for
common warehouse noise sources:
- Forklift and machinery noise
- HVAC and ventilation
- Conveyor belt hum
- Crowd/conversation babble
- Impact sounds (drops, crashes)

Features:
- Real-time spectral subtraction
- Adaptive noise estimation
- Voice Activity Detection (VAD)
- Automatic Gain Control (AGC)
- High-pass filtering for rumble removal
- Multi-band noise suppression
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from typing import Optional, Tuple, Dict, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
import time
import logging
import json
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoiseType(Enum):
    """Common warehouse noise types with frequency profiles."""
    MACHINERY = "machinery"       # 50-500 Hz continuous
    FORKLIFT = "forklift"        # 100-2000 Hz with harmonics
    HVAC = "hvac"                # 50-200 Hz low rumble
    CONVEYOR = "conveyor"        # 200-1000 Hz periodic
    IMPACT = "impact"            # Broadband transient
    CROWD = "crowd"              # 300-3000 Hz babble
    GENERAL = "general"          # Adaptive for unknown noise


@dataclass
class NoiseProfile:
    """Frequency-domain noise profile for spectral subtraction."""
    noise_spectrum: np.ndarray       # Average noise magnitude spectrum
    noise_phase: np.ndarray          # Noise phase (for advanced methods)
    sample_rate: int = 16000
    fft_size: int = 512
    update_count: int = 0
    min_frames: int = 10             # Minimum frames for reliable estimate
    
    @property
    def is_ready(self) -> bool:
        return self.update_count >= self.min_frames
    
    def update(self, spectrum: np.ndarray, alpha: float = 0.98):
        """Update noise estimate with exponential averaging."""
        if self.update_count == 0:
            self.noise_spectrum = spectrum.copy()
        else:
            self.noise_spectrum = alpha * self.noise_spectrum + (1 - alpha) * spectrum
        self.update_count += 1


@dataclass
class FilterConfig:
    """Configuration for noise filtering pipeline."""
    # General settings
    sample_rate: int = 16000
    frame_size: int = 512           # Samples per frame (~32ms at 16kHz)
    hop_size: int = 256             # Frame overlap (50%)
    
    # High-pass filter (rumble removal)
    highpass_enabled: bool = True
    highpass_cutoff: float = 80.0   # Hz
    
    # Spectral subtraction
    spectral_floor: float = 0.1     # Minimum spectral floor (0-1)
    oversubtraction: float = 2.0    # Oversubtraction factor (1-4)
    noise_estimation_time: float = 0.5  # Initial noise estimation (seconds)
    
    # Voice Activity Detection
    vad_enabled: bool = True
    vad_threshold: float = 0.3      # Energy threshold (0-1)
    vad_min_speech: float = 0.1     # Minimum speech duration (seconds)
    vad_min_silence: float = 0.3    # Minimum silence duration (seconds)
    
    # Automatic Gain Control
    agc_enabled: bool = True
    agc_target: float = 0.7         # Target RMS level (0-1)
    agc_attack: float = 0.01        # Attack time (seconds)
    agc_release: float = 0.1        # Release time (seconds)
    agc_max_gain: float = 10.0      # Maximum gain factor
    
    # Multi-band processing
    multiband_enabled: bool = True
    band_edges: list = field(default_factory=lambda: [200, 500, 1000, 2000, 4000])
    
    # Noise type specific
    noise_type: NoiseType = NoiseType.GENERAL
    
    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
            else:
                result[key] = value
        return result


class HighPassFilter:
    """
    High-pass filter for removing low-frequency rumble.
    Uses Butterworth filter design for smooth frequency response.
    """
    
    def __init__(self, cutoff: float, sample_rate: int, order: int = 4):
        self.cutoff = cutoff
        self.sample_rate = sample_rate
        self.order = order
        
        # Design Butterworth filter
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        self.b, self.a = signal.butter(order, normalized_cutoff, btype='high')
        
        # Initial filter state
        self.zi = signal.lfilter_zi(self.b, self.a)
        self.zi_state = None
        
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply high-pass filter to audio frame."""
        if self.zi_state is None:
            self.zi_state = self.zi * audio[0] if len(audio) > 0 else self.zi
            
        filtered, self.zi_state = signal.lfilter(
            self.b, self.a, audio, zi=self.zi_state
        )
        return filtered
    
    def reset(self):
        """Reset filter state."""
        self.zi_state = None


class SpectralSubtraction:
    """
    Spectral subtraction noise reduction.
    
    Estimates noise spectrum during silence and subtracts it from
    speech frames to enhance SNR.
    """
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self.fft_size = config.frame_size
        self.hop_size = config.hop_size
        
        # Hann window for STFT
        self.window = signal.windows.hann(self.fft_size)
        
        # Noise profile
        self.noise_profile = NoiseProfile(
            noise_spectrum=np.zeros(self.fft_size // 2 + 1),
            noise_phase=np.zeros(self.fft_size // 2 + 1),
            sample_rate=config.sample_rate,
            fft_size=self.fft_size
        )
        
        # Synthesis overlap-add buffer
        self.output_buffer = np.zeros(self.fft_size)
        
    def update_noise_estimate(self, frame: np.ndarray):
        """Update noise estimate from a noise-only frame."""
        if len(frame) < self.fft_size:
            frame = np.pad(frame, (0, self.fft_size - len(frame)))
            
        windowed = frame * self.window
        spectrum = np.abs(fft(windowed)[:self.fft_size // 2 + 1])
        self.noise_profile.update(spectrum)
        
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction to a frame."""
        if len(frame) < self.fft_size:
            frame = np.pad(frame, (0, self.fft_size - len(frame)))
            
        # STFT analysis
        windowed = frame * self.window
        spectrum = fft(windowed)
        magnitude = np.abs(spectrum[:self.fft_size // 2 + 1])
        phase = np.angle(spectrum[:self.fft_size // 2 + 1])
        
        # Spectral subtraction with oversubtraction
        if self.noise_profile.is_ready:
            subtracted = magnitude - self.config.oversubtraction * self.noise_profile.noise_spectrum
            
            # Apply spectral floor
            floor = self.config.spectral_floor * magnitude
            subtracted = np.maximum(subtracted, floor)
        else:
            subtracted = magnitude
            
        # Reconstruct spectrum (mirror for real signal)
        enhanced_spectrum = np.zeros(self.fft_size, dtype=complex)
        enhanced_spectrum[:self.fft_size // 2 + 1] = subtracted * np.exp(1j * phase)
        enhanced_spectrum[self.fft_size // 2 + 1:] = np.conj(enhanced_spectrum[1:self.fft_size // 2][::-1])
        
        # ISTFT synthesis
        enhanced = np.real(ifft(enhanced_spectrum))
        
        return enhanced[:len(frame)]


class VoiceActivityDetector:
    """
    Voice Activity Detection (VAD) using energy and zero-crossing rate.
    
    Detects speech segments to:
    - Enable noise estimation during silence
    - Improve ASR by providing clean start/end points
    - Reduce processing of non-speech audio
    """
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self.sample_rate = config.sample_rate
        self.frame_size = config.frame_size
        
        # Energy statistics
        self.energy_history = deque(maxlen=50)  # ~1.6 seconds at 32ms frames
        self.energy_floor = 0.0
        self.energy_ceiling = 1.0
        
        # State tracking
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speech = False
        
        # Thresholds
        self.min_speech_frames = int(config.vad_min_speech * config.sample_rate / config.frame_size)
        self.min_silence_frames = int(config.vad_min_silence * config.sample_rate / config.frame_size)
        
    def _calculate_energy(self, frame: np.ndarray) -> float:
        """Calculate normalized RMS energy."""
        rms = np.sqrt(np.mean(frame ** 2))
        return rms
    
    def _calculate_zcr(self, frame: np.ndarray) -> float:
        """Calculate zero-crossing rate."""
        signs = np.sign(frame)
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        return crossings / len(frame)
    
    def _update_statistics(self, energy: float):
        """Update energy statistics for adaptive thresholding."""
        self.energy_history.append(energy)
        
        if len(self.energy_history) >= 10:
            energies = np.array(self.energy_history)
            self.energy_floor = np.percentile(energies, 10)
            self.energy_ceiling = np.percentile(energies, 90)
            
    def process(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detect voice activity in frame.
        
        Returns:
            (is_speech, confidence)
        """
        energy = self._calculate_energy(frame)
        zcr = self._calculate_zcr(frame)
        
        self._update_statistics(energy)
        
        # Normalize energy
        energy_range = max(self.energy_ceiling - self.energy_floor, 1e-6)
        normalized_energy = (energy - self.energy_floor) / energy_range
        normalized_energy = np.clip(normalized_energy, 0, 1)
        
        # Speech typically has moderate ZCR (100-400 crossings/sec)
        # High ZCR = unvoiced speech or noise
        # Low ZCR = silence or low-frequency noise
        optimal_zcr = 0.15  # Typical for voiced speech
        zcr_weight = 1.0 - abs(zcr - optimal_zcr) / 0.3
        zcr_weight = np.clip(zcr_weight, 0.5, 1.0)
        
        # Combined confidence
        confidence = normalized_energy * zcr_weight
        
        # Decision with hysteresis
        if self.is_speech:
            if confidence < self.config.vad_threshold * 0.7:  # Lower threshold to exit
                self.silence_frames += 1
                self.speech_frames = 0
                if self.silence_frames >= self.min_silence_frames:
                    self.is_speech = False
            else:
                self.silence_frames = 0
        else:
            if confidence > self.config.vad_threshold:
                self.speech_frames += 1
                self.silence_frames = 0
                if self.speech_frames >= self.min_speech_frames:
                    self.is_speech = True
            else:
                self.speech_frames = 0
                
        return self.is_speech, confidence
    
    def reset(self):
        """Reset VAD state."""
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speech = False
        self.energy_history.clear()


class AutomaticGainControl:
    """
    Automatic Gain Control (AGC) for consistent audio levels.
    
    Normalizes audio to a target level with smooth attack/release
    to avoid pumping artifacts.
    """
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self.sample_rate = config.sample_rate
        
        # Convert times to smoothing factors
        frame_time = config.frame_size / config.sample_rate
        self.attack_coeff = 1.0 - np.exp(-frame_time / config.agc_attack)
        self.release_coeff = 1.0 - np.exp(-frame_time / config.agc_release)
        
        # Current gain
        self.gain = 1.0
        
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply AGC to audio frame."""
        # Calculate current RMS
        rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
        
        # Calculate desired gain
        desired_gain = self.config.agc_target / rms
        desired_gain = np.clip(desired_gain, 1.0 / self.config.agc_max_gain, 
                               self.config.agc_max_gain)
        
        # Smooth gain changes
        if desired_gain < self.gain:
            # Attack (compression) - fast
            self.gain += self.attack_coeff * (desired_gain - self.gain)
        else:
            # Release (expansion) - slow
            self.gain += self.release_coeff * (desired_gain - self.gain)
            
        return frame * self.gain
    
    def reset(self):
        """Reset AGC state."""
        self.gain = 1.0


class MultiBandProcessor:
    """
    Multi-band noise suppression.
    
    Applies different suppression levels to different frequency bands,
    optimized for preserving speech intelligibility.
    """
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self.sample_rate = config.sample_rate
        self.band_edges = config.band_edges
        
        # Create bandpass filters for each band
        self.filters = []
        nyquist = config.sample_rate / 2
        
        # Low band: 0 - band_edges[0]
        edges = [0] + config.band_edges + [nyquist]
        
        for i in range(len(edges) - 1):
            low = max(edges[i], 20) / nyquist  # Avoid DC
            high = min(edges[i + 1], nyquist - 10) / nyquist
            
            if low < high:
                b, a = signal.butter(4, [low, high], btype='band')
                self.filters.append({
                    'b': b, 'a': a,
                    'zi': None,
                    'low': edges[i],
                    'high': edges[i + 1]
                })
                
        # Band-specific suppression factors (0-1, 1 = no suppression)
        # Optimized to preserve speech frequencies (300-3400 Hz)
        self.band_gains = self._get_noise_type_gains(config.noise_type)
        
    def _get_noise_type_gains(self, noise_type: NoiseType) -> list:
        """Get band gains optimized for specific noise types."""
        # Default gains preserve speech bands
        defaults = [0.3, 0.7, 1.0, 0.9, 0.5, 0.3]  # Reduces low/high freq more
        
        gains = {
            NoiseType.MACHINERY: [0.2, 0.5, 0.9, 0.95, 0.7, 0.5],
            NoiseType.FORKLIFT: [0.3, 0.4, 0.85, 0.9, 0.6, 0.4],
            NoiseType.HVAC: [0.1, 0.3, 0.95, 1.0, 0.8, 0.6],
            NoiseType.CONVEYOR: [0.4, 0.5, 0.85, 0.9, 0.7, 0.5],
            NoiseType.IMPACT: [0.5, 0.6, 0.8, 0.85, 0.6, 0.4],
            NoiseType.CROWD: [0.5, 0.7, 0.75, 0.8, 0.6, 0.4],
            NoiseType.GENERAL: defaults
        }
        
        return gains.get(noise_type, defaults)
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply multi-band processing."""
        output = np.zeros_like(frame)
        
        for i, filt in enumerate(self.filters):
            if filt['zi'] is None:
                filt['zi'] = signal.lfilter_zi(filt['b'], filt['a']) * frame[0]
                
            band_audio, filt['zi'] = signal.lfilter(
                filt['b'], filt['a'], frame, zi=filt['zi']
            )
            
            gain = self.band_gains[i] if i < len(self.band_gains) else 1.0
            output += band_audio * gain
            
        return output
    
    def reset(self):
        """Reset filter states."""
        for filt in self.filters:
            filt['zi'] = None


class NoiseFilter:
    """
    Complete noise filtering pipeline for warehouse ASR.
    
    Combines multiple techniques for optimal speech enhancement:
    1. High-pass filtering (rumble removal)
    2. Spectral subtraction (stationary noise)
    3. Multi-band processing (frequency-specific suppression)
    4. Voice Activity Detection (noise estimation timing)
    5. Automatic Gain Control (level normalization)
    
    Example Usage:
        >>> config = FilterConfig(
        ...     sample_rate=16000,
        ...     noise_type=NoiseType.FORKLIFT
        ... )
        >>> filter = NoiseFilter(config)
        >>> 
        >>> # Process audio chunks
        >>> for chunk in audio_stream:
        ...     enhanced = filter.process(chunk)
        ...     asr.transcribe(enhanced)
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
        
        # Initialize components
        if self.config.highpass_enabled:
            self.highpass = HighPassFilter(
                self.config.highpass_cutoff,
                self.config.sample_rate
            )
        else:
            self.highpass = None
            
        self.spectral_sub = SpectralSubtraction(self.config)
        
        if self.config.vad_enabled:
            self.vad = VoiceActivityDetector(self.config)
        else:
            self.vad = None
            
        if self.config.agc_enabled:
            self.agc = AutomaticGainControl(self.config)
        else:
            self.agc = None
            
        if self.config.multiband_enabled:
            self.multiband = MultiBandProcessor(self.config)
        else:
            self.multiband = None
            
        # State
        self.frame_count = 0
        self.noise_estimation_frames = int(
            self.config.noise_estimation_time * 
            self.config.sample_rate / 
            self.config.frame_size
        )
        self.is_calibrating = True
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'speech_frames': 0,
            'silence_frames': 0,
            'snr_improvement_db': 0.0
        }
        
        logger.info(f"NoiseFilter initialized for {self.config.noise_type.value} noise")
        
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio through the noise filtering pipeline.
        
        Args:
            audio: Input audio samples (mono, float32, -1 to 1)
            
        Returns:
            Enhanced audio with reduced noise
        """
        self.frame_count += 1
        self.stats['frames_processed'] += 1
        
        # 1. High-pass filter
        if self.highpass:
            audio = self.highpass.process(audio)
            
        # 2. Voice Activity Detection
        is_speech = True
        vad_confidence = 1.0
        if self.vad:
            is_speech, vad_confidence = self.vad.process(audio)
            if is_speech:
                self.stats['speech_frames'] += 1
            else:
                self.stats['silence_frames'] += 1
                
        # 3. Noise estimation during silence/calibration
        if self.is_calibrating and self.frame_count <= self.noise_estimation_frames:
            self.spectral_sub.update_noise_estimate(audio)
            if self.frame_count >= self.noise_estimation_frames:
                self.is_calibrating = False
                logger.info("Noise calibration complete")
        elif not is_speech and self.vad:
            # Continue updating noise estimate during silence
            self.spectral_sub.update_noise_estimate(audio)
            
        # 4. Spectral subtraction
        audio = self.spectral_sub.process(audio)
        
        # 5. Multi-band processing
        if self.multiband:
            audio = self.multiband.process(audio)
            
        # 6. AGC
        if self.agc:
            audio = self.agc.process(audio)
            
        # Clip to valid range
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def reset(self):
        """Reset all filter states."""
        if self.highpass:
            self.highpass.reset()
        if self.vad:
            self.vad.reset()
        if self.agc:
            self.agc.reset()
        if self.multiband:
            self.multiband.reset()
            
        self.frame_count = 0
        self.is_calibrating = True
        self.stats = {
            'frames_processed': 0,
            'speech_frames': 0,
            'silence_frames': 0,
            'snr_improvement_db': 0.0
        }
        
    def get_vad_state(self) -> Tuple[bool, float]:
        """Get current VAD state and confidence."""
        if self.vad:
            return self.vad.is_speech, 0.0
        return True, 1.0
    
    def get_stats(self) -> dict:
        """Get processing statistics."""
        return self.stats.copy()


class RealTimeNoiseFilter:
    """
    Real-time noise filtering with threading support.
    
    Processes audio in a background thread with configurable
    buffer sizes and callback support.
    
    Example Usage:
        >>> def audio_callback(enhanced_audio):
        ...     asr_client.stream(enhanced_audio)
        >>> 
        >>> rt_filter = RealTimeNoiseFilter(
        ...     config=FilterConfig(noise_type=NoiseType.WAREHOUSE),
        ...     on_audio=audio_callback
        ... )
        >>> rt_filter.start()
        >>> 
        >>> # Feed audio from microphone
        >>> for chunk in mic.stream():
        ...     rt_filter.process(chunk)
        >>> 
        >>> rt_filter.stop()
    """
    
    def __init__(
        self,
        config: Optional[FilterConfig] = None,
        on_audio: Optional[Callable[[np.ndarray], None]] = None,
        on_vad: Optional[Callable[[bool, float], None]] = None,
        buffer_size: int = 4096
    ):
        self.config = config or FilterConfig()
        self.on_audio = on_audio
        self.on_vad = on_vad
        self.buffer_size = buffer_size
        
        # Core filter
        self.filter = NoiseFilter(config)
        
        # Threading
        self.input_queue: queue.Queue = queue.Queue(maxsize=100)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Buffer for frame-based processing
        self.frame_buffer = np.array([], dtype=np.float32)
        
    def start(self):
        """Start background processing thread."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        logger.info("Real-time noise filter started")
        
    def stop(self):
        """Stop background processing thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        logger.info("Real-time noise filter stopped")
        
    def process(self, audio: np.ndarray):
        """Add audio to processing queue."""
        try:
            self.input_queue.put_nowait(audio)
        except queue.Full:
            logger.warning("Audio buffer full, dropping frame")
            
    def _process_loop(self):
        """Background processing loop."""
        while self.running:
            try:
                audio = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            # Add to frame buffer
            self.frame_buffer = np.concatenate([self.frame_buffer, audio])
            
            # Process complete frames
            while len(self.frame_buffer) >= self.config.frame_size:
                frame = self.frame_buffer[:self.config.frame_size]
                self.frame_buffer = self.frame_buffer[self.config.hop_size:]
                
                # Process frame
                enhanced = self.filter.process(frame)
                
                # Callbacks
                if self.on_audio:
                    self.on_audio(enhanced)
                    
                if self.on_vad:
                    is_speech, confidence = self.filter.get_vad_state()
                    self.on_vad(is_speech, confidence)


# =============================================================================
# Preset Configurations for Common Warehouse Scenarios
# =============================================================================

class PresetConfigs:
    """Pre-configured settings for common warehouse noise scenarios."""
    
    @staticmethod
    def quiet_office() -> FilterConfig:
        """Low-noise office environment."""
        return FilterConfig(
            noise_type=NoiseType.GENERAL,
            highpass_cutoff=50.0,
            oversubtraction=1.2,
            vad_threshold=0.2,
            agc_target=0.6
        )
    
    @staticmethod
    def warehouse_general() -> FilterConfig:
        """General warehouse with mixed noise."""
        return FilterConfig(
            noise_type=NoiseType.GENERAL,
            highpass_cutoff=100.0,
            oversubtraction=2.0,
            vad_threshold=0.35,
            spectral_floor=0.15,
            agc_target=0.75
        )
    
    @staticmethod
    def forklift_area() -> FilterConfig:
        """Area with frequent forklift traffic."""
        return FilterConfig(
            noise_type=NoiseType.FORKLIFT,
            highpass_cutoff=120.0,
            oversubtraction=2.5,
            vad_threshold=0.4,
            spectral_floor=0.2,
            agc_target=0.8
        )
    
    @staticmethod
    def hvac_dominant() -> FilterConfig:
        """Area with strong HVAC/ventilation noise."""
        return FilterConfig(
            noise_type=NoiseType.HVAC,
            highpass_cutoff=150.0,  # Higher to cut HVAC rumble
            oversubtraction=3.0,
            vad_threshold=0.3,
            spectral_floor=0.1,
            agc_target=0.7
        )
    
    @staticmethod
    def conveyor_line() -> FilterConfig:
        """Near conveyor belt systems."""
        return FilterConfig(
            noise_type=NoiseType.CONVEYOR,
            highpass_cutoff=80.0,
            oversubtraction=2.2,
            vad_threshold=0.35,
            spectral_floor=0.15,
            multiband_enabled=True
        )
    
    @staticmethod
    def high_traffic() -> FilterConfig:
        """Busy area with people and machinery."""
        return FilterConfig(
            noise_type=NoiseType.CROWD,
            highpass_cutoff=80.0,
            oversubtraction=1.8,
            vad_threshold=0.4,  # Higher threshold for crowd
            vad_min_speech=0.15,  # Longer speech required
            spectral_floor=0.2,
            agc_enabled=True
        )


# =============================================================================
# Integration with NVIDIA Riva ASR
# =============================================================================

class RivaNoiseFilteredASR:
    """
    Wrapper that integrates noise filtering with NVIDIA Riva ASR.
    
    Provides seamless noise reduction before speech recognition.
    
    Example:
        >>> asr = RivaNoiseFilteredASR(
        ...     riva_uri="localhost:50051",
        ...     noise_preset="warehouse_general"
        ... )
        >>> 
        >>> # Stream audio with automatic noise filtering
        >>> for chunk in microphone:
        ...     text = asr.transcribe_chunk(chunk)
        ...     if text:
        ...         print(f"Recognized: {text}")
    """
    
    def __init__(
        self,
        riva_uri: str = "localhost:50051",
        language: str = "en-US",
        noise_preset: str = "warehouse_general",
        custom_config: Optional[FilterConfig] = None
    ):
        self.riva_uri = riva_uri
        self.language = language
        
        # Get preset or use custom config
        if custom_config:
            self.filter_config = custom_config
        else:
            presets = {
                "quiet_office": PresetConfigs.quiet_office,
                "warehouse_general": PresetConfigs.warehouse_general,
                "forklift_area": PresetConfigs.forklift_area,
                "hvac_dominant": PresetConfigs.hvac_dominant,
                "conveyor_line": PresetConfigs.conveyor_line,
                "high_traffic": PresetConfigs.high_traffic
            }
            self.filter_config = presets.get(noise_preset, PresetConfigs.warehouse_general)()
            
        # Initialize noise filter
        self.noise_filter = NoiseFilter(self.filter_config)
        
        # Riva client (lazy initialization)
        self._riva_client = None
        self._riva_config = None
        
        logger.info(f"RivaNoiseFilteredASR initialized with {noise_preset} preset")
        
    def _get_riva_client(self):
        """Lazy initialization of Riva client."""
        if self._riva_client is None:
            try:
                import riva.client
                auth = riva.client.Auth(uri=self.riva_uri)
                self._riva_client = riva.client.ASRService(auth)
                self._riva_config = riva.client.StreamingRecognitionConfig(
                    config=riva.client.RecognitionConfig(
                        encoding=riva.client.AudioEncoding.LINEAR_PCM,
                        sample_rate_hertz=self.filter_config.sample_rate,
                        language_code=self.language,
                        max_alternatives=1,
                        enable_automatic_punctuation=True
                    ),
                    interim_results=True
                )
            except ImportError:
                logger.error("nvidia-riva-client not installed")
                raise
        return self._riva_client, self._riva_config
    
    def transcribe_chunk(self, audio: np.ndarray) -> Optional[str]:
        """
        Transcribe a single audio chunk with noise filtering.
        
        Args:
            audio: Audio samples (float32, mono, -1 to 1)
            
        Returns:
            Transcribed text or None if no final result
        """
        # Apply noise filtering
        filtered = self.noise_filter.process(audio)
        
        # Convert to int16 for Riva
        audio_int16 = (filtered * 32767).astype(np.int16)
        
        # Get Riva client
        client, config = self._get_riva_client()
        
        # Note: In production, use streaming recognition
        # This is simplified for single-chunk demo
        def audio_generator():
            yield audio_int16.tobytes()
            
        try:
            responses = client.streaming_response_generator(
                audio_chunks=audio_generator(),
                streaming_config=config
            )
            
            for response in responses:
                for result in response.results:
                    if result.is_final:
                        return result.alternatives[0].transcript
        except Exception as e:
            logger.error(f"Riva ASR error: {e}")
            
        return None
    
    def get_filter_stats(self) -> dict:
        """Get noise filter statistics."""
        return self.noise_filter.get_stats()


# =============================================================================
# Demo and Testing
# =============================================================================

def demo_noise_filter():
    """Demonstrate noise filter on synthetic signal."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    print("=== Ambient Noise Filter Demo ===\n")
    
    # Parameters
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create synthetic "speech" (sine waves at speech frequencies)
    speech = 0.3 * np.sin(2 * np.pi * 300 * t)  # Fundamental
    speech += 0.2 * np.sin(2 * np.pi * 600 * t)  # Harmonic
    speech += 0.1 * np.sin(2 * np.pi * 1200 * t)  # Higher harmonic
    
    # Create synthetic noise
    # Low-frequency machinery rumble
    machinery_noise = 0.15 * np.sin(2 * np.pi * 60 * t)
    # High-frequency hiss
    hiss = 0.1 * np.random.randn(len(t))
    # Total noise
    noise = machinery_noise + hiss
    
    # Noisy signal
    noisy = speech + noise
    
    # Process with noise filter
    config = PresetConfigs.warehouse_general()
    filter = NoiseFilter(config)
    
    # Process in frames
    frame_size = config.frame_size
    filtered = np.zeros_like(noisy)
    
    for i in range(0, len(noisy) - frame_size, config.hop_size):
        frame = noisy[i:i + frame_size]
        enhanced = filter.process(frame)
        filtered[i:i + frame_size] = enhanced
        
    # Calculate SNR improvement
    noise_power_before = np.mean(noise ** 2)
    residual_noise = filtered - speech
    noise_power_after = np.mean(residual_noise ** 2)
    snr_improvement = 10 * np.log10(noise_power_before / (noise_power_after + 1e-10))
    
    print(f"Filter Configuration:")
    print(f"  - Noise Type: {config.noise_type.value}")
    print(f"  - High-pass Cutoff: {config.highpass_cutoff} Hz")
    print(f"  - Oversubtraction: {config.oversubtraction}")
    print(f"  - VAD Threshold: {config.vad_threshold}")
    print(f"\nResults:")
    print(f"  - SNR Improvement: {snr_improvement:.1f} dB")
    print(f"  - Frames Processed: {filter.stats['frames_processed']}")
    
    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    time_axis = np.arange(len(noisy)) / sample_rate
    
    axes[0].plot(time_axis, speech, 'g', alpha=0.7)
    axes[0].set_title('Original Speech Signal')
    axes[0].set_ylabel('Amplitude')
    
    axes[1].plot(time_axis, noise, 'r', alpha=0.7)
    axes[1].set_title('Noise (Machinery + Hiss)')
    axes[1].set_ylabel('Amplitude')
    
    axes[2].plot(time_axis, noisy, 'b', alpha=0.7)
    axes[2].set_title('Noisy Signal (Speech + Noise)')
    axes[2].set_ylabel('Amplitude')
    
    axes[3].plot(time_axis, filtered, 'm', alpha=0.7)
    axes[3].set_title(f'Filtered Signal (SNR Improvement: {snr_improvement:.1f} dB)')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig('noise_filter_demo.png', dpi=150)
    print("\nPlot saved to: noise_filter_demo.png")
    
    return snr_improvement


if __name__ == "__main__":
    # Run demo
    demo_noise_filter()
    
    # Show available presets
    print("\n=== Available Noise Presets ===")
    presets = [
        ("quiet_office", "Low-noise office environment"),
        ("warehouse_general", "General warehouse with mixed noise"),
        ("forklift_area", "Area with frequent forklift traffic"),
        ("hvac_dominant", "Area with strong HVAC/ventilation noise"),
        ("conveyor_line", "Near conveyor belt systems"),
        ("high_traffic", "Busy area with people and machinery"),
    ]
    
    for name, description in presets:
        print(f"  - {name}: {description}")
