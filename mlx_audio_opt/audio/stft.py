from typing import Union

import mlx.core as mx
import numpy as np
from mlx_whisper.audio import HOP_LENGTH, N_FFT, N_SAMPLES, hanning, mel_filters, stft

from mlx_audio_opt.audio.spectrogram import Spectrogram

__all__ = [
    "get_spectrogram",
    "reconstruct_audio_from_magnitude_and_phase",
    "magnitudes_to_log_mel_spectrogram",
]


def get_spectrogram(
    audio_series: Union[mx.array, np.ndarray],
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    pad_audio: int = N_SAMPLES,
    remove_last_frame: bool = True,
) -> Spectrogram:
    """Whisper-compliant STFT of the audio signal.

    Args:
        audio: The path to the audio file or an array of samples.
        n_fft: The number of FFT points.
        hop_length: The number of samples between frames.
        sampling_rate: The sampling rate of the audio.

    Returns:
        Spectrogram holding magnitudes and phase as (num_freq x num_frames).

    """
    if isinstance(audio_series, np.ndarray):
        audio_series = mx.array(audio_series)
    if pad_audio > 0:
        audio_series = mx.pad(audio_series, (0, pad_audio))

    window = hanning(n_fft)

    # transpose to comply with librosa format
    complex_spectrogram = stft(
        audio_series,
        window=window,
        nperseg=n_fft,
        noverlap=hop_length,
    )

    # mlx does not support angle, so we convert to numpy
    magnitudes = mx.abs(complex_spectrogram).T
    phase = np.angle(complex_spectrogram).T

    # whisper chops off the last frame!
    if remove_last_frame:
        magnitudes = magnitudes[:, :-1]
        phase = phase[:, :-1]

    return Spectrogram(magnitudes, phase)


def magnitudes_to_log_mel_spectrogram(
    magnitudes: mx.array,
    n_mels: int = 80,
):
    """Convert magnitudes to Whisper-compliant log Mel spectrogram.

    Args:
        magnitudes: The magnitudes array, shape (n_frames, n_freqs).
        n_mels: The number of Mel bands.

    Returns:
        mx.array of shape (n_frames, n_mels).

    """
    filters = mel_filters(n_mels)
    mel_spec = magnitudes.square() @ filters.T

    log_spec = mx.maximum(mel_spec, 1e-10).log10()
    log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
