## Arguments

# Default to our example .wav file
WAV_FILE ?= data/33711__acclivity__excessiveexposure.wav
# Default to a mlx whisper model
MODEL_ID ?= mlx-community/whisper-small-mlx

## Targets for virtual environments

# Sets up a virtual environment and activates it
setup:
	uv sync --group=dev

run:
	uv run python scripts/1_transcribe_audio.py --wav_file=$(WAV_FILE) --model_id ${MODEL_ID}
	uv run python scripts/2_visualize_audio.py --wav_file=$(WAV_FILE)
	uv run python scripts/3_optimize_audio.py --wav_file=$(WAV_FILE) --model_id ${MODEL_ID}
	uv run python scripts/4_modify_audio.py --wav_file=$(WAV_FILE) --model_id ${MODEL_ID}
