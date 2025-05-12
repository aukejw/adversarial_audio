import json
from pathlib import Path
from typing import Union

import fire
from mlx_audio.tts.generate import generate_audio

from mlx_audio_opt import REPO_ROOT

data_folder = REPO_ROOT / "data"
analysis_folder = REPO_ROOT / "analysis"


DEFAULT_TEXT = (
    f"The Mel scale is a perceptual scale, where equal distance in pitch "
    f"should sound equally different to human listeners."
)


def main(
    tts_model_id="mlx-community/Dia-1.6B",
    analysis_folder: Union[str, Path] = analysis_folder,
    text: str = DEFAULT_TEXT,
):
    """Generate audio files in the data folder."""

    for wav_file in Path(data_folder).glob("*.wav"):
        print(f"Voice cloning {wav_file}...")
        clone_voice(
            tts_model_id=tts_model_id,
            text=text,
            wav_file=wav_file,
            output_folder=analysis_folder / wav_file.stem / "4_generate_audio",
        )
    return


def clone_voice(
    tts_model_id: str,
    text: str,
    wav_file: Union[str, Path],
    output_folder: Union[str, Path],
):
    """Clone the voice of the given audio file.

    Args:
        tts_model_id: The model_id (e.g. HF) of the TTS model.
        text: The text to generate.
        wav_file: The path to the audio file.
        output_folder: The folder to save the generated audio.

    """
    # Load reference audio transcription
    transcription_file = output_folder / "transcription_nova-3.json"
    assert (
        transcription_file.exists()
    ), "Transcription file does not exist. Run 1_transcribe_audio.py first."

    with transcription_file.open("r") as f:
        transcription = dict(json.load(f))
    ref_text = transcription["results"]["summary"]["short"]

    generate_audio(
        text=text,
        model_path="mlx-community/Dia-1.6B",
        ref_audio=wav_file,
        ref_text=ref_text,
        stt_model=None,  # Only required if no transcription is provided
        file_prefix=output_folder / "generated_audio",
        audio_format="wav",
        sample_rate=44_100,  # Dia requires 44.1kHz audio.. will resample ref_audio!
        join_audio=False,
        verbose=True,
    )
    return


if __name__ == "__main__":
    fire.Fire(main)
