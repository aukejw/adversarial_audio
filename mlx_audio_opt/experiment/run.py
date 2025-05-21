import datetime
import json
from pathlib import Path
from typing import Optional, Union

from mlx_audio_opt.experiment.adversarial_audio_experiment import (
    AdversarialAudioExperiment,
)
from mlx_audio_opt.experiment.change_sentence_experiment import ChangeSentenceExperiment


def run_optimization_experiment(
    wav_file: Union[str, Path],
    output_folder: Union[str, Path],
    model_id: str,
    num_iterations: int,
    log_every_n: int,
    reload_audio_every_n: int,
    learning_rate: float,
    target_sentence: Optional[str] = None,
) -> Path:
    """Optimize given audio.

    If no target sentence is provided, we just aim to confuse the whisper
    model, minimizing the probability of the original transcription.

    If a target sentence is provided, we aim to push the model to
    produce that sentence in the transcription.

    Args:
        wav_file: The audio file to optimize.
        output_folder: The folder to save the results in.
        model_id: The model to use (typically a huggingface model id)
        num_iterations: The number of iterations to run.
        log_every_n: How often to log progress.
        reload_audio_every_n: How often to reload audio. More often = slower!
        learning_rate: Learning rate to use for the optimization.
        target_sentence: The target sentence to push the model to produce.
            If None, we just push the model to minimize the probability of the original.

    """
    wav_file = Path(wav_file)

    datetime_str = datetime.datetime.now().strftime("exp_%Y-%m-%d_%H-%M-%S")
    wav_file_output_folder = output_folder / datetime_str
    wav_file_output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\nOptimizing audio for {wav_file}...")
    print(f"  model_id: {model_id}")
    print(f"  num_iterations: {num_iterations}")
    print(f"  output folder: {wav_file_output_folder}")

    # Save a config for posterity (and perhaps reproducibility)
    config = dict(
        wav_file=Path(wav_file).resolve().as_posix(),
        datetime_str=datetime_str,
        model_id=model_id,
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        reload_audio_every_n=reload_audio_every_n,
        target_sentence=target_sentence,
    )
    config_file = wav_file_output_folder / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f)

    # Optimize
    kwargs = dict(
        model_id=model_id,
        wav_file=wav_file,
        output_folder=wav_file_output_folder,
    )
    if target_sentence is None:
        experiment_class = AdversarialAudioExperiment
    else:
        experiment_class = ChangeSentenceExperiment
        kwargs["target_sentence"] = target_sentence

    experiment = experiment_class(**kwargs)

    optimized_wav_file = experiment.run(
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        log_every_n=log_every_n,
        reload_audio_every_n=reload_audio_every_n,
    )
    print(f"Saved optimized audio to '{optimized_wav_file}'")
    return optimized_wav_file
