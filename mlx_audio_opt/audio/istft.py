from typing import Union

import librosa
import numpy as np
from mlx_whisper.audio import HOP_LENGTH, N_FFT

from mlx_audio_opt.audio.spectrogram import Spectrogram


def reconstruct_audio_from_spectrogram(
    spectrogram: Spectrogram,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    window: Union[str, np.ndarray] = "hann",
    length: int = None,
    **stft_kwargs,
):
    """Reconstruct audio from the given magnitudes and phase.

    Args:
        spectrogram: The spectrogram object containing the magnitudes and phase.
        n_fft: The number of FFT points.
        hop_length: The number of samples between frames.
        window: The window function to use.
        length: The length of the output audio signal.
        **stft_kwargs: Additional arguments for the STFT function.

    Returns:
        The reconstructed audio waveform.

    """
    magnitudes = spectrogram.magnitudes
    phase = spectrogram.phase

    if magnitudes.shape != phase.shape:
        raise ValueError("Magnitudes and phase must have the same shape")

    complex_spectrogram = magnitudes * np.exp(1j * phase)

    audio = librosa.istft(
        complex_spectrogram,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        length=length if length is None else (length + 10),
        center=False,
        **stft_kwargs,
    )
    # edge handling is poor and leads to huge spikes. We cut off a few samples.
    audio = audio[5:-5]
    return audio
