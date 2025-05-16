from typing import Union

import librosa
import mlx.core as mx
import numpy as np
from mlx_whisper.audio import HOP_LENGTH, N_FFT, mel_filters

from mlx_audio_opt.audio.spectrogram import Spectrogram


def reconstruct_audio_from_magnitude_and_phase(
    magnitudes: Union[mx.array, np.ndarray],
    phase: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
):
    """Reconstruct audio from the given magnitudes and phase.

    Parameters
    ----------
    magnitudes: array
        The magnitude of the STFT output. Mx arrays are assumed to be
        in Whisper format, i.e. (n_frames, n_freqs), and converted.
    phase: array, shape = (n_frames, M)
        The phase of the STFT output.
    n_fft: int
        The number of FFT points.
    hop_length: int
        The number of samples between frames.

    Returns
    -------
    np.array, shape = (*)
        The reconstructed audio waveform.

    """
    if isinstance(magnitudes, mx.array):
        assert magnitudes.T.shape == phase.shape, (
            f"magnitudes.T and phase must have the same shape, "
            f"but got {magnitudes.T.shape} and {phase.shape} "
        )
        magnitudes = Spectrogram.from_whisper(magnitudes).librosa_array

    # librosa expects (M, n_frames) shape
    complex_spectrogram = magnitudes * np.exp(1j * phase)
    complex_spectrogram = complex_spectrogram.T

    audio = librosa.istft(
        complex_spectrogram,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",  # note: assume hanning window was used for STFT
    )
    return audio


def magnitudes_to_log_mel_spectrogram(
    magnitudes: mx.array,
    n_mels: int = 80,
):
    """Convert magnitudes to Whisper-compliant log Mel spectrogram.

    Args:
        magnitudes: The magnitudes, shape (n_frames, n_freqs).
        n_mels: The number of Mel bands.

    Returns:
        mx.array of shape (n_frames, n_mels).

    """
    filters = mel_filters(n_mels)
    mel_spec = magnitudes @ filters.T

    log_spec = mx.maximum(mel_spec, 1e-10).log10()
    log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
