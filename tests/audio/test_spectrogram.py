import mlx.core as mx
import numpy as np
import pytest

from mlx_audio_opt.audio.spectrogram import Spectrogram
from mlx_audio_opt.audio.stft import get_spectrogram


@pytest.fixture
def spec():
    audio = np.sin(np.linspace(0, 2 * np.pi, 4000))
    return get_spectrogram(audio)


def test_spectrogram_init(spec: Spectrogram):
    magnitudes = np.array(spec.magnitudes)
    phase = np.array(spec.phase)
    spectrogram = Spectrogram(magnitudes, phase)
    np.testing.assert_allclose(spectrogram.magnitudes, spec.magnitudes)
    np.testing.assert_allclose(spectrogram.phase, spec.phase)

    phase = phase[:, :10]
    with pytest.raises(ValueError):
        Spectrogram(magnitudes, phase)


def test_spectrogram_properties(spec: Spectrogram):
    n_bins = 400 // 2 + 1
    n_frames = 3025
    assert spec.shape == (n_bins, n_frames)
    assert spec.num_frames == n_frames
    assert spec.num_bins == n_bins

    # Test whisper_tensor
    whisper = spec.whisper_tensor
    assert isinstance(whisper, mx.array)
    assert whisper.shape == (n_frames, n_bins)


def test_complex_spectrogram(spec: Spectrogram):
    complex_spec = spec.complex_spectrogram

    assert isinstance(complex_spec, np.ndarray)
    assert np.iscomplexobj(complex_spec)

    # Test that magnitude and phase are correctly used
    expected = np.array(spec.magnitudes * mx.exp(1j * spec.phase))
    np.testing.assert_allclose(complex_spec, expected)


def test_complex_spectrogram_without_phase(spec: Spectrogram):
    spec.phase = None
    with pytest.raises(ValueError):
        spec.complex_spectrogram


def test_trim(spec):
    trimmed_spec = spec.trim(10)
    assert trimmed_spec.num_frames == 10
    assert trimmed_spec.num_bins == spec.num_bins
    np.testing.assert_allclose(
        np.array(trimmed_spec.magnitudes), spec.magnitudes[:, :10]
    )
    np.testing.assert_allclose(np.array(trimmed_spec.phase), spec.phase[:, :10])

    with pytest.raises(ValueError):
        trimmed_spec.trim(11)
