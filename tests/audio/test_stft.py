import librosa
import mlx.core as mx
import numpy as np

from mlx_audio_opt.audio.stft import (
    get_spectrogram_magnitude_and_phase,
    reconstruct_audio_from_magnitude_and_phase,
)


def test_stft_libros_equivalence():
    audio = mx.random.uniform(-1, 1, (16000,))
    magnitudes, phase = get_spectrogram_magnitude_and_phase(
        audio=audio,
        n_fft=1000,
        hop_length=500,
    )
    librosa_complex_spectrogram = librosa.stft(
        y=np.array(audio),
        n_fft=1000,
        hop_length=500,
        pad_mode="reflect",
        window="hann",
        center=True,
    )
    magnitudes_librosa = (librosa_complex_spectrogram.real**2).T

    assert magnitudes.shape == (31 + 2, 501)
    assert magnitudes.shape == magnitudes_librosa.shape

    for frame_index in range(3, magnitudes.shape[0]):
        np.testing.assert_allclose(
            magnitudes[frame_index, :],
            magnitudes_librosa[frame_index, :],
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Frame {frame_index} does not match",
        )


def test_stft_and_back():
    # Convert audio to STFT and back
    audio = mx.random.uniform(-1, 1, (16000,))

    magnitudes, phase = get_spectrogram_magnitude_and_phase(
        audio=audio,
        n_fft=1000,
        hop_length=500,
    )

    # 16000 samples, n_fft=1000, hop_length=500 -> 31 frames + 2 half-padding
    # we chop off the last frame, leaving 32 frames
    assert magnitudes.shape == (31 + 2, 501)
    assert phase.shape == (31 + 2, 501)

    reconstructed_audio = reconstruct_audio_from_magnitude_and_phase(
        magnitudes=magnitudes,
        phase=phase,
        n_fft=1000,
        hop_length=500,
    )
    assert reconstructed_audio.shape == audio.shape

    np.testing.assert_allclose(
        audio,
        reconstructed_audio,
        rtol=1e-5,
        atol=1e-5,
    )
