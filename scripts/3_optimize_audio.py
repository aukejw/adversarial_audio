import json
from pathlib import Path
from typing import Union

import fire
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn
import numpy as np
from mlx_whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
from mlx_whisper.transcribe import ModelHolder, get_tokenizer

from mlx_audio_opt import REPO_ROOT

data_folder = REPO_ROOT / "data"
analysis_folder = REPO_ROOT / "analysis"


DEFAULT_TEXT = (
    f"The Mel scale is a perceptual scale, where equal distance in pitch "
    f"should sound equally different to human listeners."
)


def main(
    analysis_folder: Union[str, Path] = analysis_folder,
    model_id: str = "mlx-community/whisper-large-v3-turbo",
    text: str = DEFAULT_TEXT,
):
    """Generate audio files in the data folder."""

    for wav_file in Path(data_folder).glob("*.wav"):
        print(f"Voice cloning {wav_file}...")
        optimize_audio(
            model_id=model_id,
            wav_file=wav_file,
            output_folder=analysis_folder / wav_file.stem / "3_optimize_audio",
        )
    return


def optimize_audio(
    model_id: str,
    wav_file: Union[str, Path],
    output_folder: Union[str, Path],
):
    """Clone the voice of the given audio file.

    Args:
        tts_model_id: The model_id (e.g. HF) of the TTS model.
        wav_file: The path to the audio file.
        output_folder: The folder to save the generated audio.

    """
    # Load reference audio transcription
    model_id_suffix = model_id.split("/")[-1]
    transcription_file = output_folder / f"transcription_{model_id_suffix}.json"

    assert (
        transcription_file.exists()
    ), "Transcription file does not exist. Run 1_transcribe_audio.py first."

    with transcription_file.open("r") as f:
        transcription = dict(json.load(f))

    tokens = []
    for segment in transcription["segments"]:
        tokens.extend(segment["tokens"])

    print(tokens)

    dtype = mx.float16
    model = ModelHolder.get_model(model_id, dtype)
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language="en",
        task="transcribe",
    )
    sot_sequence = tokenizer.sot_sequence
    sot_tokens = list(sot_sequence)

    print(sot_tokens, tokens)

    # get mel spectrogram, pad to 30 seconds = 3000 frames
    mel = log_mel_spectrogram(str(wav_file), n_mels=model.dims.n_mels, padding=0)
    original_mel = mel
    mel = pad_or_trim(mel, length=N_FRAMES, axis=-2).astype(dtype)
    mel = mel[None, :]

    tokens: mx.array = mx.array(sot_tokens + tokens)
    tokens = mx.broadcast_to(tokens, (1, len(tokens)))

    print(f"Performing inference:")
    print(f"  Mel shape:          {mel.shape}")
    print(f"  Tokens shape:       {tokens.shape}")

    for i in range(20):
        # Compute gradient wrt the mel spectrogram
        loss_and_grad_fn = mx.value_and_grad(
            run_inference,
            argnums=0,
        )
        nll, grads = loss_and_grad_fn(mel, model, tokens)

        mel = mel + grads * 1e-2
        mx.eval(mel)
        print(f"  Iteration {i}: nll = {nll:.2f}")

        show_mel_spectrogram(
            mel=mel,
            original_mel=original_mel,
            figure_path=output_folder / f"optimized_mel_{model_id_suffix}_it{i}.png",
        )


def show_mel_spectrogram(
    mel: mx.array,
    original_mel: int,
    figure_path: Union[str, Path],
) -> None:
    # Display the optimized mel spectrogram
    original_mel = original_mel.transpose()
    original_num_frames = original_mel.shape[1]
    mel = np.array(mel[0, :original_num_frames].transpose())

    fig, axes = plt.subplots(2, 1, figsize=(10, 2 * 4))

    for spectrogram, ax in zip([mel - original_mel, mel], axes):
        ax.imshow(
            spectrogram,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="inferno",
        )
        ax.set_ylabel("Mel frequency")
        ax.set_xlabel("Time (frames)")

    plt.tight_layout()
    plt.savefig(
        figure_path,
        bbox_inches="tight",
        dpi=300,
    )
    return


def run_inference(
    mel: mx.array,
    model: mlx.nn.Module,
    tokens: mx.array,
) -> mx.array:
    """Perform inference on the model."""
    audio_features = model.encoder(mel)
    logits, kv_cache, _ = model.decoder(tokens, audio_features)

    # get logprob of the correct token
    log_probs = mlx.nn.log_softmax(logits[0], axis=-1)
    num_tokens = tokens.shape[1]
    assert log_probs.shape == (num_tokens, 51866), logits.shape

    neg_log_likelihood = log_probs[mx.arange(num_tokens), tokens[0]]
    assert neg_log_likelihood.shape == (num_tokens,), neg_log_likelihood.shape

    neg_log_likelihood = neg_log_likelihood.sum(axis=0)
    return neg_log_likelihood


if __name__ == "__main__":
    fire.Fire(main)
