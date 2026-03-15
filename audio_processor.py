"""Audio preprocessing utilities for Voice Morph."""
import os
import numpy as np
import librosa
import soundfile as sf


def load_audio(file_path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate."""
    audio, orig_sr = librosa.load(file_path, sr=sr, mono=True)
    return audio, sr


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """RMS normalize audio to target dB level."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return audio
    target_rms = 10 ** (target_db / 20.0)
    gain = target_rms / rms
    normalized = audio * gain
    # Prevent clipping
    peak = np.max(np.abs(normalized))
    if peak > 0.99:
        normalized = normalized * (0.99 / peak)
    return normalized


def trim_silence(audio: np.ndarray, sr: int = 22050,
                 top_db: int = 30) -> np.ndarray:
    """Trim leading and trailing silence."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def preprocess_audio(file_path: str, sr: int = 22050,
                     normalize: bool = True,
                     trim: bool = True) -> tuple[np.ndarray, int]:
    """Full preprocessing pipeline: load, trim silence, normalize."""
    audio, sr = load_audio(file_path, sr=sr)
    if trim:
        audio = trim_silence(audio, sr=sr)
    if normalize:
        audio = normalize_audio(audio)
    return audio, sr


def save_audio(audio: np.ndarray, file_path: str, sr: int = 22050,
               format: str = "wav"):
    """Save audio to file."""
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    sf.write(file_path, audio, sr, format=format)


def get_audio_info(file_path: str) -> dict:
    """Get audio file metadata."""
    info = sf.info(file_path)
    return {
        "duration": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
    }
