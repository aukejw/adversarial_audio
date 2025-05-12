from pathlib import Path
from typing import Dict, Union

import mlx.core as mx
import mlx_whisper
import numpy as np
from mlx_whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
from mlx_whisper.decoding import DecodingOptions, DecodingTask
from mlx_whisper.transcribe import ModelHolder


def transcribe_audio(
    wav_file: Union[str, Path],
    model_id: str = "mlx-community/whisper-large-v3-turbo",
) -> Dict:
    """Transcribe the given audio file using the specified model.

    Args:
        wav_file: The path to the audio file.
        model_id: The model_id (e.g. HF) of the STT model.

    Returns:
        A dictionary containing the transcription results. Should contain
        text (raw transcription), segments (sentences) and language (en for English).

    """
    transcription = mlx_whisper.transcribe(
        audio=str(wav_file),
        path_or_hf_repo=model_id,
        fp16=True,
        word_timestamps=True,
    )

    text = transcription["text"]
    segments = transcription["segments"]
    language = transcription["language"]

    return transcription


def get_words_from_transcription(
    transcription: Dict[str, Union[str, list]],
) -> Dict[str, Union[str, list]]:
    """Get the words from the given transcription.

    Args:
        transcription: The transcription dictionary.

    Returns:
        A dictionary containing the words and their timestamps and confidence.

    """
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


def inference(
    wav_file: Union[str, Path],
    model_id: str = "mlx-community/whisper-large-v3-turbo",
    mask_top_k: int = 0,
) -> Dict[str, Union[str, list]]:
    """Run inference with the Whisper STT model.

    Args:
        wav_file: The path to the audio file.
        model_id: The model_id (e.g. HF) of the STT model.
        mask_top_k: The number of top logits to mask during decoding.

    Returns:
        A dictionary containing the inference results.

    """
    dtype = mx.float16
    model = ModelHolder.get_model(model_id, dtype)

    # get mel spectrogram, pad to 30 seconds of frames
    mel = log_mel_spectrogram(str(wav_file), n_mels=model.dims.n_mels, padding=0)
    mel = pad_or_trim(mel, length=N_FRAMES, axis=-2).astype(dtype)
    mel = mel[None, :]

    decoding_task = DecodingTask(
        model=model,
        options=DecodingOptions(
            task="transcribe",
            language="en",
            prompt="",
            prefix="",
        ),
    )
    # Get the complete list of logits by repeating the main loop of the whisper model
    decoding_task.inference.reset()
    decoding_task.decoder.reset()

    n_audio: int = mel.shape[0]
    audio_features: mx.array = decoding_task._get_audio_features(mel)
    tokens: mx.array = mx.array(decoding_task.initial_tokens)
    tokens = mx.broadcast_to(tokens, (n_audio, len(decoding_task.initial_tokens)))

    n_batch = tokens.shape[0]
    sum_logprobs = mx.zeros(n_batch)

    def _step(inputs, audio_features, tokens, sum_logprobs):
        pre_logits = decoding_task.inference.logits(inputs, audio_features)

        # consider the logits at the last token only
        logits = pre_logits[:, -1]

        # apply the logit filters, e.g. for suppressing or applying penalty to
        for logit_filter in decoding_task.logit_filters:
            logits = logit_filter.apply(logits, tokens)

        if mask_top_k > 0:
            # get the top k logits, ascending order
            index_order = mx.argsort(logits, axis=-1)
            top_k_indices = index_order[:, -mask_top_k:]

            # mask the top k logits
            mask = np.zeros(logits.shape, np.float32)
            mask[top_k_indices] = -np.inf
            mask = mx.array(mask)

            logits = logits + mask

        # expand the tokens tensor with the selected next tokens
        tokens, completed, sum_logprobs = decoding_task.decoder.update(
            tokens, logits, sum_logprobs
        )
        return tokens, completed, sum_logprobs, pre_logits

    # initial step
    tokens, completed, sum_logprobs, pre_logits = _step(
        tokens, audio_features, tokens, sum_logprobs
    )

    # compute no_speech_probs
    if decoding_task.tokenizer.no_speech is not None:
        probs_at_sot = mx.softmax(pre_logits[:, decoding_task.sot_index], axis=-1)
        no_speech_probs = probs_at_sot[:, decoding_task.tokenizer.no_speech]
    else:
        no_speech_probs = mx.full(n_batch, mx.nan)
    mx.async_eval(completed, tokens, sum_logprobs, no_speech_probs)

    # decode
    all_token_logits = []
    for i in range(1, decoding_task.sample_len):
        inputs = tokens[:, -1:]
        next_tokens, next_completed, next_sum_logprobs, pre_logits = _step(
            inputs, audio_features, tokens, sum_logprobs
        )
        mx.async_eval(next_completed, next_tokens, next_sum_logprobs)
        if completed:
            break
        tokens = next_tokens
        completed = next_completed
        sum_logprobs = next_sum_logprobs

        all_token_logits.append(pre_logits[:, -1])

    return all_token_logits
