from typing import Union

import mlx.core as mx
import numpy as np
from IPython.display import Audio, display
from mlx_whisper.audio import SAMPLE_RATE


def display_audio_player(
    audio: Union[mx.array, np.ndarray],
    title: str,
    sample_rate: int = SAMPLE_RATE,
) -> Audio:
    """Show audio."""

    display(title)
    audio = Audio(np.array(audio), rate=sample_rate)
    display(audio)

    return audio
