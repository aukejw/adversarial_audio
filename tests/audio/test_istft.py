import mlx.core as mx
import numpy as np
import pytest
from mlx_whisper.audio import N_FRAMES, N_SAMPLES

from mlx_audio_opt.audio.istft import reconstruct_audio_from_spectrogram
from mlx_audio_opt.audio.spectrogram import Spectrogram
from mlx_audio_opt.audio.stft import get_spectrogram


@pytest.fixture
def audio():
    x = np.arange(0, 4000)
    return np.sin(x) + np.cos(x / 2) + np.random.normal(0, 0.1, x.shape)


def test_stft_and_back_unpadded(audio):
    # STFT using Whisper implementation
    spectrogram = get_spectrogram(
        audio_series=audio,
        n_fft=1000,
        hop_length=500,
        remove_last_frame=False,
        pad_audio=0,
    )

    # 4000 samples, n_fft=1000, hop_length=500 -> 7 frames + 2 half-padding
    #
    #     |__0__|__2__|__4__|__6__|_pad__|
    # |_pad__|__1__|__3__|__5__|__7__|
    #     .     .     .     .     .
    #     0   1000  2000  3000  4000
    #
    assert spectrogram.magnitudes.shape == (501, 7 + 2)
    assert spectrogram.phase.shape == (501, 7 + 2)

    # reconstruct the audio
    reconstructed_audio = reconstruct_audio_from_spectrogram(
        spectrogram=spectrogram,
        n_fft=1000,
        hop_length=500,
        length=len(audio),
    )
    assert reconstructed_audio.shape == audio.shape
    np.testing.assert_allclose(
        reconstructed_audio,
        audio,
        rtol=1e-4,
        atol=1e-4,
    )


def test_stft_and_back_padded(audio):
    # STFT using Whisper implementation and default parameters
    spectrogram = get_spectrogram(
        audio_series=audio,
        remove_last_frame=False,
    )
    # remove the resulting padding frames
    spectrogram = spectrogram.trim(spectrogram.num_frames - N_FRAMES)

    # 4000 samples, n_fft=400, hop_length=160 -> 24 frames + 2 half-padding
    assert spectrogram.magnitudes.shape == (201, 24 + 2)
    assert spectrogram.phase.shape == (201, 24 + 2)

    # reconstruct the audio
    reconstructed_audio = reconstruct_audio_from_spectrogram(
        spectrogram=spectrogram,
        length=len(audio),
    )
    assert reconstructed_audio.shape == audio.shape
    np.testing.assert_allclose(
        reconstructed_audio,
        audio,
        rtol=1e-4,
        atol=1e-4,
    )


def test_istft_and_back(audio):
    mx.random.seed(0)

    # STFT using Whisper implementation, using default padding and so on
    spectrogram = get_spectrogram(
        audio_series=audio,
        remove_last_frame=False,
        pad_audio=N_SAMPLES,  # adds 30 seconds of signal padding
    )
    # 4000 samples, n_fft=400, hop_length=160 -> 24 frames + 2 half-padding
    n_content_frames = 24 + 2
    expected_shape = (400 // 2 + 1, n_content_frames + N_FRAMES)
    assert spectrogram.magnitudes.shape == expected_shape
    assert spectrogram.phase.shape == expected_shape

    # Remove the padding frames
    spectrogram = spectrogram.trim(n_content_frames)
    magnitudes = spectrogram.magnitudes
    phase = spectrogram.phase

    # Alter the magnitudes
    new_magnitudes = magnitudes * 2
    spectrogram_altered = Spectrogram(
        magnitudes=new_magnitudes,
        phase=phase,
    )

    # reconstruct audio from altered spectrogram
    audio_recon = reconstruct_audio_from_spectrogram(
        spectrogram=spectrogram_altered,
        length=len(audio),
    )

    # apply STFT to the reconstructed audio
    spectrogram_recon = get_spectrogram(
        audio_recon,
        remove_last_frame=False,
        pad_audio=N_SAMPLES,
    )
    spectrogram_recon = spectrogram_recon.trim(n_content_frames)

    # now check that the final spectrogram makes sense
    assert spectrogram_altered.magnitudes.shape == (201, n_content_frames)
    assert spectrogram_altered.phase.shape == (201, n_content_frames)
    assert spectrogram_recon.magnitudes.shape == (201, n_content_frames)
    assert spectrogram_recon.phase.shape == (201, n_content_frames)

    np.testing.assert_allclose(
        spectrogram_recon.magnitudes,
        spectrogram_altered.magnitudes,
        rtol=1e-4,
        atol=1e-4,
    )
