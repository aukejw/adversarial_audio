import copy
from typing import Optional, Self, Union

import mlx.core as mx
import numpy as np


class Spectrogram:
    """Spectrogram class to handle conversion between libraries.

    We canonically represent a spectrogram as a 2d array, with shape
    (n_mel, n_frames), same as librosa.

    Args:
        magnitudes: Magnitudes as a 2D array (n_bins, n_frames).
        phase: Phase is a 2D array (n_bins, n_frames), can be None.

    """

    def __init__(
        self,
        magnitudes: Union[np.ndarray, mx.array],
        phase: Optional[Union[np.ndarray, mx.array]] = None,
    ):
        self.magnitudes = mx.array(magnitudes)
        self.phase = mx.array(phase)

        if phase is not None and magnitudes.shape != phase.shape:
            raise ValueError(
                f"magnitudes.shape={magnitudes.shape}), and phase.shape={phase.shape}. "
                f"These must have the same shape."
            )

    def trim(
        self,
        n_frames: int,
    ) -> Self:
        """Return a spectrogram trimmed to the specified number of frames."""
        if n_frames < self.num_frames:
            magnitudes = self.magnitudes[:, :n_frames]
            phase = self.phase[:, :n_frames] if self.phase is not None else None
        else:
            raise ValueError(
                f"n_frames={n_frames}, but this spectrogram has {self.num_frames} frames."
            )
        return Spectrogram(magnitudes, phase)

    @property
    def complex_spectrogram(self):
        if self.phase is None:
            raise ValueError("Phase information is not available.")
        return np.array(self.magnitudes * mx.exp(1j * self.phase))

    @property
    def shape(self):
        return self.magnitudes.shape

    @property
    def num_frames(self):
        return self.magnitudes.shape[1]

    @property
    def num_bins(self):
        return self.magnitudes.shape[0]

    @property
    def whisper_tensor(self):
        return copy.deepcopy(self.magnitudes.T).astype(mx.float32)
