from pathlib import Path
from typing import Union

import fire

from mlx_audio_opt import REPO_ROOT
from mlx_audio_opt.experiment.run import run_optimization_experiment

analysis_folder = REPO_ROOT / "analysis"


def main(
    wav_file: Union[str, Path],
    analysis_folder: Union[str, Path] = analysis_folder,
    model_id: str = "mlx-community/whisper-small-mlx",
    num_iterations: int = 50,
    log_every_n: int = 50,
    reload_audio_every_n: int = 5,
    learning_rate: float = 1e-2,
    l2_penalty: float = 0.0,
):
    """Adversarially perturb the given audio.

    We just aim to confuse the given model, minimizing the probability
    of outputting the original transcription.

    """
    wav_file = Path(wav_file)
    assert wav_file.exists(), f"File '{wav_file}' does not exist."

    output_folder = analysis_folder / wav_file.stem / "3_optimize_audio"

    run_optimization_experiment(
        wav_file=wav_file,
        output_folder=output_folder,
        model_id=model_id,
        num_iterations=num_iterations,
        log_every_n=log_every_n,
        reload_audio_every_n=reload_audio_every_n,
        learning_rate=learning_rate,
        l2_penalty=l2_penalty,
    )


if __name__ == "__main__":
    fire.Fire(main)
