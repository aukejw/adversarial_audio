from typing import Tuple, Union

import librosa
import mlx.core as mx
import numpy as np
from mlx_whisper.audio import HOP_LENGTH, N_FFT, hanning, mel_filters, stft

from mlx_audio_opt.audio.spectrogram import Spectrogram


def get_phase_and_magnitudes(
    audio_series: Union[mx.array, np.ndarray],
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Whisper-compliant STFT of the audio signal.

    Args:
        audio: The path to the audio file or an array of samples.
        n_fft: The number of FFT points.
        hop_length: The number of samples between frames.
        sampling_rate: The sampling rate of the audio.

    Returns:
        magnitudes: The magnitudes (power=1).
        phase: The phase, obtained using np.angle.

    """
    if isinstance(audio_series, np.ndarray):
        audio_seriesj = mx.array(audio_series)

    window = hanning(n_fft)

    # transpose to comply with librosa format
    complex_spectrogram = stft(
        audio_series,
        window=window,
        nperseg=n_fft,
        noverlap=hop_length,
    ).T

    # mlx does not support angle and direct conversion to np complex fails,
    # so we convert to numpy in a roundabout way
    real = np.array(mx.real(complex_spectrogram))
    imag = np.array(mx.imag(complex_spectrogram))
    complex_spectrogram = real + 1j * imag

    magnitudes = np.abs(complex_spectrogram)
    phase = np.angle(complex_spectrogram)
    return magnitudes, phase


def reconstruct_audio_from_magnitude_and_phase(
    magnitudes: Union[mx.array, np.ndarray],
    phase: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    window: Union[str, np.ndarray] = "hann",
    **stft_kwargs,
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
        window=window,
        **stft_kwargs,
    )
    return audio


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
    mel_spec = magnitudes @ filters.T

    log_spec = mx.maximum(mel_spec, 1e-10).log10()
    log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
