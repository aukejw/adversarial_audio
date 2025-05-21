import librosa
import mlx.core as mx
import mlx_whisper.audio
import numpy as np
import pytest
from mlx_whisper.audio import HOP_LENGTH, N_FFT, hanning, stft

from mlx_audio_opt.audio.stft import get_spectrogram, magnitudes_to_log_mel_spectrogram


@pytest.fixture
def audio():
    x = np.arange(0, 4_000)
    return np.sin(x) + np.cos(x / 2) + np.random.normal(0, 0.1, x.shape)


def test_stft_libros_equivalence(audio):
    spectrogram = get_spectrogram(
        audio_series=audio,
        n_fft=1000,
        hop_length=500,
        pad_audio=0,
        remove_last_frame=False,
    )

    librosa_complex_spectrogram = librosa.stft(
        y=np.array(audio),
        n_fft=1000,
        hop_length=500,
        pad_mode="reflect",
        window="hann",
        center=True,
    )
    magnitudes_librosa = np.abs(librosa_complex_spectrogram)
    phase_librosa = np.angle(librosa_complex_spectrogram)

    # 4000 samples, n_fft=1000, hop_length=500 -> 7 frames + 2 half-padding
    #
    #     |__0__|__2__|__4__|__6__|_pad__|
    # |_pad__|__1__|__3__|__5__|__7__|
    #     .     .     .     .     .
    #     0   1000  2000  3000  4000
    #
    # whisper does chop off the last padding frame, but we've disabled this here
    n_frames = 7 + 2
    assert magnitudes_librosa.shape == (501, n_frames)
    assert phase_librosa.shape == (501, n_frames)
    assert spectrogram.magnitudes.shape == (501, n_frames)
    assert spectrogram.phase.shape == (501, n_frames)

    np.testing.assert_allclose(
        spectrogram.magnitudes,
        magnitudes_librosa,
        rtol=1e-4,
        atol=1e-4,
    )

    real_librosa = np.real(librosa_complex_spectrogram)
    imag_librosa = np.imag(librosa_complex_spectrogram)

    complex_spectrogram = spectrogram.magnitudes * mx.exp(
        1j * mx.array(spectrogram.phase)
    )
    real_ours = np.array(mx.real(complex_spectrogram))
    imag_ours = np.array(mx.imag(complex_spectrogram))

    np.testing.assert_allclose(
        real_librosa,
        real_ours,
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        imag_librosa,
        imag_ours,
        rtol=1e-4,
        atol=1e-4,
    )


def test_log_mel_spectrogram():
    # ground truth: Whispers implementation
    audio = mx.random.uniform(-1, 1, (16000,))

    # whisper
    window = hanning(N_FFT)
    freqs = stft(audio, window, nperseg=N_FFT, noverlap=HOP_LENGTH)
    magnitudes_whisper = freqs[:-1, :].abs().T

    # mlx_audio_opt
    spectrogram = get_spectrogram(
        audio_series=audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        pad_audio=0,
        remove_last_frame=True,
    )
    assert spectrogram.magnitudes.shape == magnitudes_whisper.shape
    assert spectrogram.phase.shape == magnitudes_whisper.shape

    np.testing.assert_allclose(
        spectrogram.magnitudes,
        magnitudes_whisper,
        atol=1e-7,
        rtol=1e-7,
    )

    # whisper log mel spectrogram
    log_mel_spectrogram_whisper = mlx_whisper.audio.log_mel_spectrogram(
        audio=audio,
        n_mels=80,
        padding=0,
    )
    assert log_mel_spectrogram_whisper.shape == (100, 80)

    # mlx_audio_opt log mel spectrogram
    magnitudes_array = mx.array(spectrogram.magnitudes).T
    log_mel_spectrogram_mx = magnitudes_to_log_mel_spectrogram(
        magnitudes=magnitudes_array,
        n_mels=80,
    )
    assert log_mel_spectrogram_mx.shape == log_mel_spectrogram_whisper.shape
    np.testing.assert_allclose(
        log_mel_spectrogram_whisper,
        log_mel_spectrogram_mx,
        rtol=1e-3,
        atol=1e-5,
    )
