from pathlib import Path
from typing import Union

import fire
import matplotlib.pyplot as plt

from mlx_audio_opt import REPO_ROOT
from mlx_audio_opt.visualization.plot_audio import visualize_audio

data_folder = REPO_ROOT / "data"
output_folder = REPO_ROOT / "analysis"


def main(
    data_folder: Union[str, Path] = data_folder,
    output_folder: Union[str, Path] = output_folder,
):
    """Analyze audio files in the data folder.

    Args:
        data_folder: The folder containing the audio files.
        output_folder: The folder to save the analysis results to.

    """
    for wav_file in Path(data_folder).glob("*.wav"):
        print(f"Visualizing {wav_file}...")

        wav_file_output_folder = output_folder / wav_file.stem / "2_visualize_audio"
        wav_file_output_folder.mkdir(parents=True, exist_ok=True)

        output_file = wav_file_output_folder / "original_audio.png"
        if output_file.exists():
            print(f"Already visualized {wav_file}. Skipping...")
            continue

        kwargs = dict(wav_file=wav_file)

        for transcription_name in [
            "transcription_nova-3.json",
            "transcription_whisper-large-v3-turbo.json",
        ]:
            transcription_path = wav_file_output_folder / transcription_name
            if transcription_path.exists():
                kwargs[transcription_path.name] = transcription_path
                print(f"  found transcription '{transcription_path}'")

        fig = visualize_audio(**kwargs)

        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        print(f"Saved figure to '{output_file}'")
        plt.close()

    return


if __name__ == "__main__":
    fire.Fire(main)
