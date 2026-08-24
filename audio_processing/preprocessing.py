"""
Audio cleaning pipeline: resample, high-pass filter, denoise, and
loudness-normalize before diarization/transcription.
"""

import numpy as np
import soundfile as sf
import librosa
import os
import pyloudnorm as pyln
import shutil
from scipy.signal import butter, filtfilt

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False


def prepare_diarization_audio(input_path, output_path, target_sample_rate=16000):
    """Create a mono, 16 kHz, PCM-16 WAV from the unenhanced input audio.

    This conversion intentionally performs only channel downmixing, resampling,
    and PCM encoding. It does not denoise, normalize, filter, or separate
    vocals, preserving the original signal characteristics for pyannote.
    """
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    source_info = sf.info(input_path)
    already_compatible = (
        source_info.format == "WAV"
        and source_info.channels == 1
        and source_info.samplerate == target_sample_rate
        and source_info.subtype == "PCM_16"
    )
    if already_compatible:
        if os.path.abspath(input_path) != os.path.abspath(output_path):
            shutil.copyfile(input_path, output_path)
    else:
        audio, sample_rate = sf.read(input_path, always_2d=True, dtype="float32")
        if audio.shape[1] == 1:
            mono = audio[:, 0]
        else:
            mono = np.mean(audio, axis=1, dtype=np.float32)

        if sample_rate != target_sample_rate:
            mono = librosa.resample(
                mono,
                orig_sr=sample_rate,
                target_sr=target_sample_rate,
            )

        mono = np.clip(mono, -1.0, 1.0).astype(np.float32, copy=False)
        sf.write(
            output_path,
            mono,
            target_sample_rate,
            format="WAV",
            subtype="PCM_16",
        )

    prepared_info = sf.info(output_path)
    if not (
        prepared_info.format == "WAV"
        and prepared_info.channels == 1
        and prepared_info.samplerate == target_sample_rate
        and prepared_info.subtype == "PCM_16"
    ):
        raise RuntimeError(
            "Prepared diarization audio does not satisfy WAV/mono/16 kHz/PCM-16 requirements"
        )

    print(f"Diarization audio: {output_path}")
    print(f"Channels: {prepared_info.channels}")
    print(f"Sample rate: {prepared_info.samplerate} Hz")
    print("Format: PCM 16-bit WAV")
    return output_path


def load_and_resample(path, target_sr=16000):
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, sr


def high_pass_filter(audio, sr, cutoff=80.0):
    """Removes low-frequency rumble/hum that can confuse VAD and diarization."""
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(2, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, audio)


def denoise(audio, sr):
    """Spectral-gating noise reduction. Skips gracefully if the package is missing."""
    if not HAS_NOISEREDUCE:
        print("noisereduce not installed, skipping denoise step.")
        return audio
    return nr.reduce_noise(y=audio, sr=sr, stationary=False)


def normalize_loudness(audio, sr, target_lufs=-23.0):
    """Normalizes to a consistent integrated loudness (EBU R128 style)."""
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    return pyln.normalize.loudness(audio, loudness, target_lufs)


def clean_audio(
    input_path,
    output_path,
    target_sr=16000,
    do_highpass=True,
    highpass_cutoff=80.0,
    do_denoise=True,
    do_normalize=True,
    target_lufs=-23.0,
):
    print(f"Cleaning audio: {input_path}")

    audio, sr = load_and_resample(input_path, target_sr)

    if do_highpass:
        audio = high_pass_filter(audio, sr, highpass_cutoff)

    if do_denoise:
        audio = denoise(audio, sr)

    if do_normalize:
        audio = normalize_loudness(audio, sr, target_lufs)

    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

    sf.write(output_path, audio, sr)
    print(f"Cleaned audio saved: {output_path}")

    return output_path
