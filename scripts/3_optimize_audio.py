from pathlib import Path
from typing import Union

import fire

from mlx_audio_opt import REPO_ROOT
from mlx_audio_opt.experiment import AdversarialAudioExperiment

analysis_folder = REPO_ROOT / "analysis"


def main(
    wav_file: Union[str, Path],
    analysis_folder: Union[str, Path] = analysis_folder,
    model_id: str = "mlx-community/whisper-small-mlx",
    num_iterations: int = 100,
    learning_rate: float = 1e-2,
):
    """Optimize the given audio."""

    wav_file = Path(wav_file)
    short_model_id = model_id.split("/")[-1]

    wav_file_output_folder = analysis_folder / wav_file.stem / "3_optimize_audio"
    wav_file_output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\nOptimizing audio for {wav_file}...")
    print(f"  model_id: {model_id}")
    print(f"  num_iterations: {num_iterations}")
    print(f"  output folder: {wav_file_output_folder}")

    # Optimize
    experiment = AdversarialAudioExperiment(
        model_id=model_id,
        wav_file=wav_file,
        output_folder=wav_file_output_folder,
    )
    optimized_wav_file = experiment.run(
        num_iterations=num_iterations,
        learning_rate=learning_rate,
    )
    print(f"Saved optimized audio to '{optimized_wav_file}'")
    return


if __name__ == "__main__":
    fire.Fire(main)
