from pathlib import Path
from typing import Dict, Optional, Union

import librosa
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

from mlx_audio_opt.stt.transcription import Transcription
from mlx_audio_opt.visualization.plot_transcription import plot_transcription_words

__all__ = [
    "visualize_audio",
    "compare_spectrograms",
]


def save_and_close(
    fig: plt.Figure,
    output_file: Union[str, Path],
) -> None:
    """Save and close the figure.

    Args:
        fig: The matplotlib figure.
        output_file: The file path.

    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close(fig)


def visualize_audio(
    wav_file: Union[str, Path],
    sampling_rate: Optional[int],
    **transcriptions: Dict[str, Transcription],
) -> plt.Figure:
    """Visualize the audio data.

    Args:
        wav_file: The path to the audio file.
        sampling_rate: The sampling rate of the audio file.
        transcription: The transcriptoins or a path to their files.

    Returns:
        The matplotlib figure.

    """
    audio_series, sampling_rate = librosa.load(
        wav_file,
        sr=sampling_rate,
    )

    num_rows = 2 + len(transcriptions)
    fig, axes = plt.subplots(num_rows, 1, figsize=(10, num_rows * 3))
    fig.suptitle(f"Audio file {wav_file.name}")

    # Plot original waveform
    axes[0].set_title("Waveform")
    axes[0].grid(True)

    librosa.display.waveshow(
        audio_series,
        sr=sampling_rate,
        ax=axes[0],
    )

    # Plot mel spectrogram, some default settings from librosa
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_series,
        sr=sampling_rate,
        n_fft=2048,  # length of FFT window (win_length defaults to this number)
        hop_length=512,  # number of samples between frames (determines overlap!)
        n_mels=128,  # number of Mel bands
    )
    axes[1].set_title("Mel Spectrogram")
    img = librosa.display.specshow(
        librosa.power_to_db(mel_spectrogram, ref=np.max),
        sr=sampling_rate,
        x_axis="time",
        y_axis="mel",
        ax=axes[1],
    )
    fig.colorbar(img, ax=axes[1], format="%+2.f dB")

    # Plot transcription as barchart with confidence scores
    for index, (name, transcription) in enumerate(transcriptions.items()):
        plot_transcription_words(
            ax=axes[2 + index],
            title=name,
            words=transcription.get_words(),
        )

    # Clear up the plots
    num_seconds = audio_series.shape[0] / sampling_rate
    for ax in axes:
        ax.set_xlim(-0.5, num_seconds + 0.5)
        xticks = range(0, int(num_seconds) + 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.set_xlabel("Time (seconds)")

    fig.tight_layout()
    return fig


def compare_spectrograms(
    magnitudes1: mx.array,
    magnitudes2: mx.array,
    title1: str,
    title2: str,
    ylabel: str,
    suptitle: str,
) -> plt.Figure:
    """Create a visual comparison of two spectrogram-like inputs.

    Args:
        magnitudes1: The first spectrogram magnitudes (power=1).
        magnitudes2: The second spectrogram magnitudes (power=1).
        title1: The title for the first spectrogram.
        title2: The title for the second spectrogram.
        ylabel: The label for the y-axis.
        suptitle: The title for the entire figure.

    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 3 * 4))
    fig.suptitle(suptitle)

    for spectrogram, ax, title in [
        (magnitudes1, axes[0], title1),
        (magnitudes2, axes[1], title2),
        (magnitudes1 - magnitudes2, axes[2], "Difference"),
    ]:
        image = ax.imshow(
            spectrogram,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="inferno",
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time (frames)")
        fig.colorbar(image, ax=ax)

    fig.tight_layout()
    return fig
