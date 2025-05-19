import librosa
import mlx.core as mx
import mlx_whisper.audio
import numpy as np
from mlx_whisper.audio import HOP_LENGTH, N_FFT, hanning, stft

from mlx_audio_opt.audio.istft import reconstruct_audio_from_spectrogram
from mlx_audio_opt.audio.stft import get_spectrogram, magnitudes_to_log_mel_spectrogram


def test_stft_libros_equivalence():
    audio = mx.random.uniform(-1, 1, (16000,))
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
    magnitudes_librosa = np.abs(librosa_complex_spectrogram.real)

    assert spectrogram.magnitudes.shape == (501, 32 + 1)
    assert spectrogram.magnitudes.shape == magnitudes_librosa.shape

    np.testing.assert_allclose(
        spectrogram.magnitudes,
        magnitudes_librosa,
        rtol=1e-3,
        atol=40,  # the difference can be quite large - not really equivalent settings
    )


def test_stft_and_back():
    # Convert audio to STFT and back
    audio = mx.random.uniform(-1, 1, (16000,))

    spectrogram = get_spectrogram(
        audio_series=audio,
        n_fft=1000,
        hop_length=500,
        pad_audio=0,
    )

    # 16000 samples, n_fft=1000, hop_length=500 -> 31 frames + 2 half-padding
    # we chop off the last frame, leaving 32 frames
    assert spectrogram.magnitudes.shape == (501, 31 + 1)
    assert spectrogram.phase.shape == (501, 31 + 1)

    reconstructed_audio = reconstruct_audio_from_spectrogram(
        spectrogram=spectrogram,
        n_fft=1000,
        hop_length=500,
        length=len(audio),
    )
    assert reconstructed_audio.shape == audio.shape

    # we can't really test much more here - the match is not great
    # because the librosa istft does not match Whispers stft operation.


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
