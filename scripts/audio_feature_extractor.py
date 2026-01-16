"""
Audio Feature Extractor for Paper Evaluation Pipeline

Extracts 12D chroma features at 30 Hz to match the lighting data frame rate.
Used for computing SSM correlation between audio and lighting in Table 3.

Configuration (matching alternator_v1.3):
    - sampling_rate: 22050 Hz
    - hop_length: 735 samples (22050 / 30 = 735 for 30 Hz output)
    - window_size: 2940 samples (n_fft)

Reference:
    - alternator_v1.3/Sources/Preprocessing/Audio/audio_extractor.py
    - alternator_v1.3/RunScriptsAndConfigs/Preprocessing/Audio/audio_extraction.conf
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Audio extraction configuration matching alternator_v1.3 standard config."""

    sampling_rate: int = 22050
    hop_length: int = 735  # 22050 / 30 = 735 for 30 Hz
    window_size: int = 2940  # n_fft

    @property
    def frame_rate(self) -> float:
        """Output frame rate in Hz."""
        return self.sampling_rate / self.hop_length


class AudioFeatureExtractor:
    """
    Extract audio features for SSM correlation with lighting.

    Primary output: 12D chroma_stft features at 30 Hz.

    Usage:
        extractor = AudioFeatureExtractor()
        chroma = extractor.extract_chroma(audio_path)
        # chroma.shape = (frames, 12) at 30 Hz
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file at target sampling rate."""
        audio, sr = librosa.load(audio_path, sr=self.config.sampling_rate)
        return audio, sr

    def extract_chroma_stft(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract STFT-based chroma features from audio.

        Following alternator_v1.3 approach:
        1. Apply HPSS to get harmonic component (cleaner chroma)
        2. Compute chroma_stft on harmonic audio

        Args:
            audio: Audio waveform at self.config.sampling_rate

        Returns:
            chroma: Shape (frames, 12) at 30 Hz - 12D chroma features
        """
        # Separate harmonic and percussive components
        # Using harmonic component gives cleaner pitch content for chroma
        audio_harmonic, _ = librosa.effects.hpss(audio)

        # Extract chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio_harmonic,
            sr=self.config.sampling_rate,
            hop_length=self.config.hop_length,
            n_fft=self.config.window_size,
            n_chroma=12,
        )

        # Transpose to (frames, 12) format
        return chroma.T

    def extract_chroma_cqt(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract CQT-based chroma features (alternative to STFT).

        CQT provides logarithmically-spaced frequency bins which can be
        better for music with varying pitch content.

        Args:
            audio: Audio waveform at self.config.sampling_rate

        Returns:
            chroma: Shape (frames, 12) at 30 Hz
        """
        audio_harmonic, _ = librosa.effects.hpss(audio)

        chroma = librosa.feature.chroma_cqt(
            y=audio_harmonic,
            sr=self.config.sampling_rate,
            hop_length=self.config.hop_length,
            n_chroma=12,
        )

        return chroma.T

    def extract_chroma(self, audio_path: str, method: str = "stft") -> np.ndarray:
        """
        Extract chroma features from audio file.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            method: 'stft' or 'cqt' - chroma extraction method

        Returns:
            chroma: Shape (frames, 12) at 30 Hz
        """
        audio, _ = self.load_audio(audio_path)

        if method == "stft":
            return self.extract_chroma_stft(audio)
        elif method == "cqt":
            return self.extract_chroma_cqt(audio)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'stft' or 'cqt'.")

    def extract_onset_beat(self, audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """Extract binary beat array for thesis-aligned beat alignment."""
        _, audio_percussive = librosa.effects.hpss(audio)

        tempo, beat_frames = librosa.beat.beat_track(
            y=audio_percussive,
            sr=self.config.sampling_rate,
            hop_length=self.config.hop_length,
        )

        if hasattr(tempo, "__len__") and len(tempo) > 0:
            tempo_val = float(tempo[0])
        else:
            tempo_val = float(tempo)

        total_frames = int(len(audio) / self.config.hop_length) + 1
        onset_beat = np.zeros(total_frames, dtype=np.float32)

        for bf in beat_frames:
            if bf < total_frames:
                onset_beat[bf] = 1.0

        return onset_beat, tempo_val

    def extract_all_features(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive audio features matching alternator_v1.3.

        Returns dict with:
            - chroma_stft: (frames, 12)
            - chroma_cqt: (frames, 12)
            - onset_env: (frames,)
            - onset_beat: (frames,) - binary beat array for beat alignment
            - rms: (frames,)
            - tempo: scalar
        """
        audio, sr = self.load_audio(audio_path)
        audio_harmonic, audio_percussive = librosa.effects.hpss(audio)

        features = {}

        features["chroma_stft"] = self.extract_chroma_stft(audio)
        features["chroma_cqt"] = self.extract_chroma_cqt(audio)

        onset_env = librosa.onset.onset_strength(
            y=audio_percussive,
            sr=self.config.sampling_rate,
            hop_length=self.config.hop_length,
            aggregate=np.median,
        )
        features["onset_env"] = onset_env

        onset_beat, tempo_val = self.extract_onset_beat(audio)
        features["onset_beat"] = onset_beat
        features["tempo"] = tempo_val

        S, _ = librosa.magphase(
            librosa.stft(
                audio, hop_length=self.config.hop_length, n_fft=self.config.window_size
            )
        )
        rms = librosa.feature.rms(
            S=S, hop_length=self.config.hop_length, frame_length=self.config.window_size
        )
        features["rms"] = rms.flatten()

        return features


def get_hop_length_for_fps(sample_rate: int, target_fps: int = 30) -> int:
    """
    Calculate hop_length to achieve target frame rate.

    Formula: hop_length = sample_rate / target_fps

    Example:
        22050 Hz / 30 fps = 735 samples per hop
        44100 Hz / 30 fps = 1470 samples per hop
    """
    return int(sample_rate / target_fps)


if __name__ == "__main__":
    import yaml

    # Load paths config
    config_path = Path(__file__).parent.parent / "configs" / "paths.yaml"
    with open(config_path) as f:
        paths = yaml.safe_load(f)

    audio_dir = Path(paths["audio_parts"])

    # Test extraction on first audio file
    extractor = AudioFeatureExtractor()

    sample_files = list(audio_dir.glob("*.wav"))[:3]

    print(f"Audio Feature Extractor")
    print(f"=======================")
    print(
        f"Config: sr={extractor.config.sampling_rate}, hop={extractor.config.hop_length}"
    )
    print(f"Output frame rate: {extractor.config.frame_rate} Hz")
    print()

    for audio_file in sample_files:
        print(f"Processing: {audio_file.name}")

        try:
            chroma = extractor.extract_chroma(str(audio_file))
            print(f"  Chroma shape: {chroma.shape}")
            print(f"  Duration: {chroma.shape[0] / 30:.2f}s at 30 Hz")
            print(f"  Chroma range: [{chroma.min():.3f}, {chroma.max():.3f}]")
        except Exception as e:
            print(f"  Error: {e}")

        print()
