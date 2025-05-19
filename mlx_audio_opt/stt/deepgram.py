import os
from pathlib import Path
from typing import Union

from dotenv import load_dotenv

from deepgram import DeepgramClient, FileSource, PrerecordedOptions
from mlx_audio_opt.stt.transcription import DeepgramTranscription

__all__ = [
    "transcribe_audio",
]


def transcribe_audio(
    audio: Union[str, Path],
    model_id: str = "nova-3",
) -> DeepgramTranscription:
    """Transcribe audio data using the Deepgram API.

    Args:
        audio: The audio, as file or as array.
        model_id: The model_id (e.g. HF) of the STT model.

    Returns:
        A transcription result. For the format, also see
        https://developers.deepgram.com/reference/speech-to-text-api/listen

    """
    if isinstance(audio, (str, Path)):
        audio = Path(audio)
        with open(audio, "rb") as source:
            buffer_data = source.read()

    else:
        raise TypeError(f"audio must be str, Path, but got {type(audio)}")

    load_dotenv()
    deepgram = DeepgramClient(
        api_key=os.environ["DEEPGRAM_API_KEY"],
    )
    options = PrerecordedOptions(
        model=model_id,
        smart_format=True,
        summarize="v2",
    )
    payload: FileSource = {
        "buffer": buffer_data,
    }
    response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
    response = response.to_dict()

    return DeepgramTranscription(response)
