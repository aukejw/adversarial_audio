import json
from pathlib import Path
from typing import Any, Dict, List, Union

import librosa
import matplotlib.pyplot as plt
import numpy as np

import mlx_audio_opt.stt.deepgram
import mlx_audio_opt.stt.whisper


def visualize_audio(
    wav_file: Union[str, Path],
    **transcription_files: Dict[str, Path],
) -> plt.Figure:
    """Visualize the audio data.

    Args:
        wav_file: The path to the audio file.
        transcription_files: The path to transcription files.

    Returns:
        The matplotlib figure.

    """
    audio_series, sampling_rate = librosa.load(wav_file, sr=None)

    num_rows = 2 + len(transcription_files)
    fig, axes = plt.subplots(num_rows, 1, figsize=(10, num_rows * 3))
    fig.suptitle(f"Original audio of {wav_file.name}")

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
    librosa.display.specshow(
        librosa.power_to_db(mel_spectrogram, ref=np.max),
        sr=sampling_rate,
        x_axis="time",
        y_axis="mel",
        ax=axes[1],
    )

    # Plot transcription as barchart with confidence scores
    for i, (name, transcription_file) in enumerate(transcription_files.items()):
        ax = axes[2 + i]
        ax.set_title(name)
        ax.grid(True)

        with transcription_file.open("r") as f:
            transcription = json.load(f)

        if "nova" in transcription_file.name:
            transcribed_words = mlx_audio_opt.stt.deepgram.get_words_from_transcription(
                transcription
            )
        elif "whisper" in transcription_file.name:
            transcribed_words = mlx_audio_opt.stt.whisper.get_words_from_transcription(
                transcription
            )
        else:
            raise NotImplementedError(
                f"Not sure how to handle transcription '{transcription_file.name}'"
            )

        plot_transcription(ax=ax, words=transcribed_words)

    # Clear up the plots
    num_seconds = audio_series.shape[0] / sampling_rate
    for ax in axes:
        ax.set_xlim(-0.5, num_seconds + 0.5)
        xticks = range(0, int(num_seconds) + 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.set_xlabel("Time (seconds)")

    plt.tight_layout()
    return fig


def plot_transcription(
    ax: plt.Axes,
    words: List[Dict[str, Any]],
) -> None:
    """Plot the words with confidence scores.

    Args:
        ax: The matplotlib axes to plot on.
        words: List of word predictions.

    """
    for word_dict in words:
        word_start: float = word_dict["start"]
        word_end: float = word_dict["end"]
        word_confidence: float = word_dict["confidence"]
        punctuated_word: str = word_dict["word"].strip()

        # Show the words
        bar_width = word_end - word_start
        x_center = word_start + bar_width / 2
        ax.bar(
            x=x_center,
            height=word_confidence,
            width=bar_width,
            align="center",
            alpha=0.7,
        )

        # Add rotated word label
        ax.text(
            x=x_center,
            y=0.02,
            s=punctuated_word,
            rotation=90,
            ha="center",
            va="bottom",
        )

    ax.set_ylim(0, 1.1)  # Set y-axis limit for confidence (0-1)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Confidence")
    return
