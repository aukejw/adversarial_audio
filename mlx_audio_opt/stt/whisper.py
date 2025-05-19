from pathlib import Path
from typing import Union

import mlx.core as mx
import mlx_whisper
import mlx_whisper.whisper
from mlx_whisper.audio import N_FRAMES, pad_or_trim

from mlx_audio_opt.audio.stft import magnitudes_to_log_mel_spectrogram
from mlx_audio_opt.stt.transcription import WhisperTranscription

__all__ = [
    "transcribe_audio",
    "get_log_probabilities",
]


def transcribe_audio(
    audio: Union[str, Path, mx.array],
    model_id: str = "mlx-community/whisper-small-mlx",
    fp16: bool = True,
    word_timestamps: bool = True,
    **kwargs,
) -> WhisperTranscription:
    """Transcribe the given audio file using the specified model.

    Args:
        wav_file: The path to the audio file.
        model_id: The model_id (e.g. HF) of the STT model.

    Returns:
        A transcription result.

    """
    if isinstance(audio, (str, Path)):
        if not Path(audio).exists():
            raise ValueError(f"Audio file '{audio}' does not exist")
        audio = Path(audio).as_posix()

    transcription = mlx_whisper.transcribe(
        audio=audio,
        path_or_hf_repo=model_id,
        fp16=fp16,
        word_timestamps=word_timestamps,
        task="transcribe",
        **kwargs,
    )
    return WhisperTranscription(transcription)


def get_log_probabilities(
    model: mlx_whisper.whisper.Whisper,
    magnitudes: mx.array,
    tokens: mx.array,
    n_frames: int = N_FRAMES,
):
    """Get the logprobs of the given tokens for given audio.

    Args:
        model: The Whisper model.
        magnitudes: The magnitudes of the audio.
        tokens: The tokens to get the logprobs for.

    Returns:
        The logprobs of the given tokens.

    """
    # mlx_whisper does something interesting:
    # 1. pad the audio to N_SAMPLES
    # 2. compute the STFT on the padded audio
    # 3. convert magnitudes to log mel spectrogram
    # 4. crop the mel spectrogram (usually back to pre-padding n_frames)
    # 5. then pad the mel spectrogram to N_FRAMES
    #
    # We assume we're at step 3 here
    mel = magnitudes_to_log_mel_spectrogram(
        magnitudes,
        n_mels=model.dims.n_mels,
    )

    # first trim back to original audio length, discarding audio padding
    n_content_frames = mel.shape[-2] - n_frames
    n_content_frames = min(n_frames, n_content_frames)
    mel = mel[0:n_content_frames]

    # then pad the log mel spectrogram to n_frames
    mel = pad_or_trim(mel, n_frames, axis=-2)
    mel = mel.astype(mx.float16)

    if mel.ndim == 2:
        mel = mel[None]

    audio_features = model.encoder(mel)
    logits, kv_cache, _ = model.decoder(tokens, audio_features)

    # stable log softmax
    logits = logits.astype(mx.float32)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    # Quick sanity check
    log_probs = mx.squeeze(log_probs)

    b, num_tokens = tokens.shape
    vocab_size = model.dims.n_vocab
    assert b == 1, "We can only handle batch size 1 here"
    assert log_probs.shape == (num_tokens, vocab_size), log_probs.shape

    # select the log probabilities of the token sequence, p(tokens|audio)
    # the first entry is the probability of the second token
    # the third entry is the probability of the fourth token
    # etc
    next_tokens = tokens[0, 1:]
    log_probs = log_probs[mx.arange(num_tokens - 1), next_tokens]
    # we assume the first token is always start-of-sentence here, with probability 1
    log_probs = mx.concat((mx.zeros((1,), dtype=log_probs.dtype), log_probs), axis=0)
    assert log_probs.shape == (
        num_tokens,
    ), f"log_probs.shape={log_probs.shape}, but must be ({num_tokens},)"

    return log_probs
