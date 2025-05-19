from typing import Optional, Self

import mlx.core as mx
import numpy as np


class Spectrogram:
    """Spectrogram class to handle conversion between libraries.

    We canonically represent a spectrogram as a 2d array, with shape
    (n_mel, n_frames), same as librosa.

    """

    def __init__(
        self,
        magnitudes: np.ndarray,
        phase: Optional[np.ndarray] = None,
    ):
        self.magnitudes = magnitudes
        self.phase = phase

    def trim(self, n_frames: int) -> Self:
        """Trim the spectrogram to the specified number of frames."""
        if n_frames < self.num_frames:
            magnitudes = self.magnitudes[:, :n_frames]
            phase = self.phase[:, :n_frames] if self.phase is not None else None
        else:
            raise ValueError(
                f"n_frames={n_frames}, but this spectrogram has {self.num_frames} frames."
            )
        return Spectrogram(magnitudes, phase)

    @property
    def shape(self):
        return self.magnitudes.shape

    @property
    def num_frames(self):
        return self.magnitudes.shape[1]

    @property
    def num_frequencies(self):
        return self.magnitudes.shape[0]

    @property
    def librosa_array(self):
        return self.magnitudes

    @property
    def whisper_tensor(self):
        return mx.array(self.magnitudes.T)

    @classmethod
    def from_librosa(cls, spectrogram: np.ndarray):
        return cls(spectrogram)

    @classmethod
    def from_whisper(cls, spectrogram: mx.array):
        return cls(np.array(spectrogram.T))
