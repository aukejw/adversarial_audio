from typing import Any, Dict, List

import matplotlib.pyplot as plt

from mlx_audio_opt.stt.transcription import Transcription

__all__ = [
    "plot_transcriptions",
    "plot_transcription_words",
]


def plot_transcriptions(
    suptitle: str,
    transcriptions: Dict[str, Transcription],
    xlim: List[float] = None,
) -> plt.Figure:
    """Plot given transcriptions."""

    num_rows = len(transcriptions)
    fig, axes = plt.subplots(num_rows, 1, figsize=(8, 2.5 * num_rows), sharex=True)
    fig.suptitle(suptitle, fontsize=14)

    for ax, (name, transcription) in zip(axes, transcriptions.items()):
        ax.grid(True)
        plot_transcription_words(
            words=transcription.get_words(),
            title=f"Transcription '{name}'",
            ax=ax,
            xlim=xlim,
        )

    fig.tight_layout()
    fig.show()
    return fig


def plot_transcription_words(
    ax: plt.Axes,
    title: str,
    words: List[Dict[str, Any]],
    xlim: List[float] = None,
    title_fontsize: int = 14,
) -> None:
    """Plot the words with confidence scores.

    Args:
        ax: The matplotlib axes to plot on.
        words: List of word predictions.

    """
    ax.grid(True)
    ax.set_title(
        title if title else " ",
        fontdict={"fontsize": title_fontsize},
    )

    for word_dict in words:
        word_start: float = word_dict["start"]
        word_end: float = word_dict["end"]
        word_confidence: float = word_dict["confidence"]
        punctuated_word: str = word_dict["word"].strip()

        if word_end == word_start:
            # ensure a bar is always visible
            word_start -= 0.01

        # Show the words as distinctly colored bars, height = confidence
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
            fontdict={
                "fontsize": 12,
            },
        )

    ax.set_ylim(0, 1.00)  # Set y-axis limit for confidence (0-1)
    ax.set_xlabel(
        "Time (seconds)",
        fontdict={"fontsize": title_fontsize},
    )
    ax.set_ylabel(
        "Confidence",
        fontdict={"fontsize": title_fontsize},
    )

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
        if xlim[1] - xlim[0] > 1:
            ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))

    return
