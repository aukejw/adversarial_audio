import os
from pathlib import Path
from typing import Any, Dict, Union

from dotenv import load_dotenv

from deepgram import DeepgramClient, FileSource, PrerecordedOptions

__all__ = [
    "transcribe_audio",
]


def transcribe_audio(
    wav_file: Union[str, Path],
    model_id: str = "nova-3",
) -> Dict[str, Any]:
    """Transcribe audio data using the Deepgram API.

    Args:
        wav_file: The path to the audio file.
        model_id: The model_id (e.g. HF) of the STT model.

    Returns:
        A dictionary containing the transcription results.
        For the format, see
        https://developers.deepgram.com/reference/speech-to-text-api/listen

    """
    wav_file = Path(wav_file)

    load_dotenv()
    deepgram = DeepgramClient(
        api_key=os.environ["DEEPGRAM_API_KEY"],
    )
    options = PrerecordedOptions(
        model=model_id,
        smart_format=True,
        summarize="v2",
    )
    with open(wav_file, "rb") as source:
        buffer_data = source.read()

    payload: FileSource = {
        "buffer": buffer_data,
    }
    with open(wav_file, "rb") as source:
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
    response = response.to_dict()

    return response
