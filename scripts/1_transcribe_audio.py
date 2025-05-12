import json
from pathlib import Path
from typing import Dict, Union

import fire

import mlx_audio_opt.stt.deepgram
import mlx_audio_opt.stt.whisper
from mlx_audio_opt import REPO_ROOT

data_folder = REPO_ROOT / "data"
output_folder = REPO_ROOT / "analysis"


def main(
    data_folder: Union[str, Path] = data_folder,
    output_folder: Union[str, Path] = output_folder,
    model_id: str = "nova-3",
):
    """Analyze audio files in the data folder.

    Args:
        data_folder: The folder containing the audio files.
        output_folder: The folder to save the transcriptions to.
        model_type: Type of model to use for transcription.
            For example, options "nova-3" or "whisper-large-v3-turbo".

    """
    for wav_file in Path(data_folder).glob("*.wav"):
        print(f"Analyzing {wav_file}...")

        wav_file_output_folder = output_folder / wav_file.stem / "1_transcribe_audio"
        output_file = wav_file_output_folder / f"transcription_{model_id}.json"
        if output_file.exists():
            print(f"Transcription already exists: {output_file}")
            continue

        if "nova" in model_id:
            response: Dict = mlx_audio_opt.stt.deepgram.transcribe_audio(
                wav_file=wav_file,
                model_id=model_id,
            )
        elif "whisper" in model_id:
            if not model_id.startswith("mlx_community/"):
                model_id = f"mlx-community/{model_id}"
            response: Dict = mlx_audio_opt.stt.whisper.transcribe_audio(
                wav_file=wav_file,
                model_id=f"mlx-community/{model_id}",
            )
        else:
            raise NotImplementedError(f"Model_id '{model_id}' not implemented")

        wav_file_output_folder.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(response, f, indent=4)

        print(f"Transcription saved to '{output_file}'")

    return


if __name__ == "__main__":
    fire.Fire(main)
