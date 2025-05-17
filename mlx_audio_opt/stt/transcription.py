from pathlib import Path
from typing import Any, Dict, List, Union

from mlx_audio_opt.file_io import try_load_json

__all__ = [
    "Transcription",
    "WhisperTranscription",
    "DeepgramTranscription",
    "get_transcription",
]


class Transcription:
    """Base transcription container."""

    def __init__(self, transcription: Dict[str, Any]):
        self.transcription = transcription

    def get_text(self) -> str:
        """Get the transcribed text."""
        raise NotImplementedError

    def get_words(self) -> List[Dict]:
        """Get words. Each word dict contains 'word', 'start', 'end', and 'confidence'."""
        raise NotImplementedError

    def get_tokens(self) -> List[int]:
        """Get the tokens from the transcript."""
        raise NotImplementedError


class WhisperTranscription(Transcription):
    def get_text(self) -> str:
        return self.transcription["text"]

    def get_words(self) -> List[Dict]:
        transcribed_words = []
        for segment in self.transcription["segments"]:
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

    def get_tokens(self) -> List[int]:
        transcribed_tokens = []
        for segment in self.transcription["segments"]:
            transcribed_tokens.extend(segment["tokens"])
        return transcribed_tokens


class DeepgramTranscription(Transcription):
    def get_text(self) -> str:
        summary = self.transcription["results"]["summary"]
        assert summary["result"] == "success", summary
        return summary["short"]

    def get_words(self) -> List[Dict]:
        channel = self.transcription["results"]["channels"][0]
        words = channel["alternatives"][0]["words"]

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


def get_transcription(
    transcription: Union[str, Path, Dict[str, Any]],
) -> Transcription:
    """Get a transcription container.

    Args:
        transcription: The transcription dictionary.

    Returns:
        The container.

    """
    transcription = try_load_json(transcription)

    if "results" in transcription and "channels" in transcription["results"]:
        transcription = DeepgramTranscription(transcription)

    elif "text" in transcription and "segments" in transcription:
        transcription = WhisperTranscription(transcription)

    else:
        raise NotImplementedError("We could not recognize the transcription type.")

    return transcription
