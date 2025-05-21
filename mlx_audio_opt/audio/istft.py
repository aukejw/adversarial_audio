import warnings
from typing import Union

import librosa
import mlx.core as mx
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
    magnitudes = np.array(spectrogram.magnitudes, dtype=np.float32)
    phase = np.array(spectrogram.phase, dtype=np.float32)

    if magnitudes.shape != phase.shape:
        raise ValueError("Magnitudes and phase must have the same shape")

    complex_spectrogram = magnitudes * mx.exp(1j * mx.array(phase))
    complex_spectrogram = np.array(complex_spectrogram)

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)

        original_fftlib = librosa.core.get_fftlib()

        class mx_fftlib:
            """Custom fft lib-like object that implements irfft"""

            @staticmethod
            def irfft(x, n=None, axis=-1):
                return np.array(
                    mx.fft.irfft(
                        mx.array(x),
                        n=n,
                        axis=axis,
                    )
                )

        librosa.core.set_fftlib(mx_fftlib)

    audio = librosa.istft(
        complex_spectrogram,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        length=length,
        center=True,
        **stft_kwargs,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        librosa.core.set_fftlib(original_fftlib)

    return audio
