## Targets for virtual environments
WAV_FILE ?= data/33711__acclivity__excessiveexposure.wav

# Sets up a virtual environment and activates it
setup:
	uv sync --group=dev

run:
	uv run python scripts/1_transcribe_audio.py --wav_file=$(WAV_FILE)
	uv run python scripts/2_visualize_audio.py --wav_file=$(WAV_FILE)
	uv run python scripts/3_optimize_audio.py --wav_file=$(WAV_FILE)
