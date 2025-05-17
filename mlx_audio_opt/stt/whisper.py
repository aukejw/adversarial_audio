from pathlib import Path
from typing import Dict, Union

import mlx.core as mx
import mlx_whisper
import mlx_whisper.whisper

from mlx_audio_opt.audio.stft import magnitudes_to_log_mel_spectrogram

__all__ = [
    "transcribe_audio",
]


def transcribe_audio(
    wav_file: Union[str, Path],
    model_id: str = "mlx-community/whisper-small-mlx",
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


def get_log_probabilities(
    model: mlx_whisper.whisper.Whisper,
    magnitudes: mx.array,
    tokens: mx.array,
):
    """Get the logprobs of the given tokens for given audio.

    Args:
        model: The Whisper model.
        magnitudes: The magnitudes of the audio.
        tokens: The tokens to get the logprobs for.

    Returns:
        The logprobs of the given tokens.

    """
    mel = magnitudes_to_log_mel_spectrogram(
        magnitudes**2,
        n_mels=model.dims.n_mels,
    )

    # encode audio, decode logits
    mel = mel[None]
    audio_features = model.encoder(mel)
    logits, kv_cache, _ = model.decoder(tokens, audio_features)

    # stable log softmax
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    # Quick sanity check
    log_probs = mx.squeeze(log_probs)
    num_tokens = tokens.shape[1]
    vocab_size = model.dims.n_vocab
    assert log_probs.shape == (num_tokens, vocab_size), log_probs.shape

    # select the log probabilities of the token sequence, p(tokens|audio)
    log_probs = log_probs[mx.arange(num_tokens), tokens[0]]
    assert log_probs.shape == (num_tokens,), log_probs.shape

    return log_probs
