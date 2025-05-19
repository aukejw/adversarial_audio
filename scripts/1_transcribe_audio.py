from pathlib import Path
from typing import Union

import fire

from mlx_audio_opt import REPO_ROOT
from mlx_audio_opt.file_io import to_json
from mlx_audio_opt.stt import deepgram, whisper
from mlx_audio_opt.stt.transcription import DeepgramTranscription, WhisperTranscription

OUTPUT_FOLDER = REPO_ROOT / "analysis"


def main(
    wav_file: Union[str, Path],
    output_folder: Union[str, Path] = OUTPUT_FOLDER,
    model_id: str = "nova-3",
):
    """Analyze audio files in the data folder.

    Args:
        wav_file: The path to the audio file to transcribe.
        output_folder: The folder to save the transcriptions to.
        model_id: Identifier for model to use for transcription.
            Example options include "nova2", "nova-3", "whisper-small-mlx"

    """
    wav_file = Path(wav_file)
    output_folder = Path(output_folder)

    print(f"Transcribing {wav_file}...")
    print(f"  model_id: {model_id}")

    if model_id.startswith("mlx-community/"):
        model_id = model_id.replace("mlx-community/", "")

    wav_file_output_folder = output_folder / wav_file.stem / "1_transcribe_audio"
    output_file = wav_file_output_folder / f"transcription_{model_id}.json"

    if output_file.exists():
        print(f"Transcription already exists: {output_file}")
        return output_file

    if "nova" in model_id:
        transcription: DeepgramTranscription = deepgram.transcribe_audio(
            wav_file=wav_file,
            model_id=model_id,
        )

    elif "whisper" in model_id:
        transcription: WhisperTranscription = whisper.transcribe_audio(
            wav_file=wav_file,
            model_id=f"mlx-community/{model_id}",
        )

    else:
        raise NotImplementedError(f"Model_id '{model_id}' not implemented")

    print(f"Transcription: {transcription.get_text()}")

    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    to_json(
        dictionary=transcription.transcription,
        output_file=output_file,
        verbose=True,
    )
    return output_file


if __name__ == "__main__":
    fire.Fire(main)
