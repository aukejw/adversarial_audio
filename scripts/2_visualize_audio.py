from pathlib import Path
from typing import Union

import fire
import matplotlib.pyplot as plt

from mlx_audio_opt import REPO_ROOT
from mlx_audio_opt.stt.transcription import get_transcription
from mlx_audio_opt.visualization.plot_audio import visualize_audio

output_folder = REPO_ROOT / "analysis"


def main(
    wav_file: Union[str, Path],
    output_folder: Union[str, Path] = output_folder,
):
    """Analyze audio files in the data folder.

    Args:
        wav_file: The path to the audio file to analyze.
        output_folder: The folder to save the analysis results to.

    """
    wav_file = Path(wav_file)
    output_folder = Path(output_folder)

    print(f"\nVisualizing {wav_file}...")

    wav_file_output_folder = output_folder / wav_file.stem / "2_visualize_audio"
    wav_file_output_folder.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        wav_file=wav_file,
        sampling_rate=None,
    )

    transcription_folder = wav_file_output_folder.parent / "1_transcribe_audio"
    for transcription_path in transcription_folder.glob("transcription_*.json"):
        name = transcription_path.stem.replace("transcription_", "")
        kwargs[name] = get_transcription(transcription_path)
        print(f"  found transcription '{transcription_path}'")

    visualize_audio(**kwargs)

    output_file = wav_file_output_folder / "original_audio.png"
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    print(f"Saved figure to '{output_file}'")
    plt.close()


if __name__ == "__main__":
    fire.Fire(main)
