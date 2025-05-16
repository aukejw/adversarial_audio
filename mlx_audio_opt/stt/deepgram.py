from pathlib import Path
from typing import Any, Dict, Union

from deepgram import DeepgramClient, FileSource, PrerecordedOptions


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

    deepgram = DeepgramClient()
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


def get_words_from_transcription(
    transcription: Dict[str, Any],
) -> Dict[str, Any]:
    """Get the words from the given transcription.

    Args:
        transcription: The transcription dictionary.

    Returns:
        A list containing the words and their timestamps and confidence.

    """
    words = transcription["results"]["channels"][0]["alternatives"][0]["words"]

    transcribed_words = []
    for word in words:
        transcribed_words.append(
            dict(
                word=word["punctuated_word"],
                confidence=word["confidence"],
                start=word["start"],
                end=word["end"],
            )
        )
    return transcribed_words
