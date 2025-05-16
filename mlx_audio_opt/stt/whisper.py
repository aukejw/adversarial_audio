from pathlib import Path
from typing import Any, Dict, List, Union

import mlx_whisper
from mlx_whisper.tokenizer import Tokenizer

from mlx_audio_opt.file_io import try_load_json

__all__ = [
    "transcribe_audio",
    "get_text_from_transcription",
    "get_words_from_transcription",
    "get_tokens_from_transcription",
]


def transcribe_audio(
    wav_file: Union[str, Path],
    model_id: str = "mlx-community/whisper-large-v3-turbo",
    fp16: bool = True,
    word_timestamps: bool = True,
    **kwargs,
) -> Dict:
    """Transcribe the given audio file using the specified model.

    Args:
        wav_file: The path to the audio file.
        model_id: The model_id (e.g. HF) of the STT model.

    Returns:
        A dictionary containing the transcription results. Should contain
        text (raw transcription), segments (sentences) and language (en for English).

    """
    assert Path(wav_file).exists(), f"Audio file '{wav_file}' does not exist"

    transcription = mlx_whisper.transcribe(
        audio=str(wav_file),
        path_or_hf_repo=model_id,
        fp16=fp16,
        word_timestamps=word_timestamps,
        task="transcribe",
        **kwargs,
    )
    return transcription


def get_text_from_transcription(
    transcription: Union[Dict[str, Any], str, Path],
) -> Dict[str, Union[str, list]]:
    """Get the text from the given transcription.

    Args:
        transcription: The transcription dictionary.

    Returns:
        A dictionary containing the text and its segments.

    """
    transcription = try_load_json(transcription)
    return transcription["text"]


def get_words_from_transcription(
    transcription: Union[Dict[str, Any], str, Path],
) -> Dict[str, Union[str, list]]:
    """Get the words from the given transcription.

    Args:
        transcription: The transcription dictionary.

    Returns:
        A dictionary containing the words and their timestamps and confidence.

    """
    transcription = try_load_json(transcription)

    transcribed_words = []
    for segment in transcription["segments"]:
        for word in segment["words"]:
            transcribed_words.append(
                dict(
                    word=word["word"],
                    start=word["start"],
                    end=word["end"],
                    confidence=word["probability"],
                )
            )
    return transcribed_words


def get_tokens_from_transcription(
    tokenizer: Tokenizer,
    transcription: Union[Dict[str, Any], str, Path],
    add_eot_token: bool = True,
) -> List[int]:
    """Get the tokens from the given transcription.

    Args:
        transcription: The transcription dictionary.

    Returns:
        A list of tokens for each segment (including sot, eot).

    """
    transcription = try_load_json(transcription)

    transcribed_tokens = []
    for segment in transcription["segments"]:
        transcribed_tokens.extend(segment["tokens"])

    # add sot and eot tokens
    tokens = list(tokenizer.sot_sequence) + transcribed_tokens
    if add_eot_token:
        tokens = tokens + [tokenizer.eot]

    return tokens
