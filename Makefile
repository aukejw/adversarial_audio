## Targets for virtual environments

# Sets up a virtual environment and activates it
setup:
	uv python install 3.11.11
	uv sync --group=dev
