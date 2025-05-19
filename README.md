# MLX speech optimization

Using pretrained speech models to investigate speech data.

## Installation

You will need:

  - Apple silicon
  - `uv` for dependency management
  - `ffmpeg` for `mlx_whisper` transcriptions
  - a Deepgram API key (add yours to a `.env` file) for Deepgram transcriptions
  - one or more `.wav` files

1. Add your Deepgram API key to a `.env` file:
   ```
   cp .env.template .env
   sed -i 's/DEEPGRAM_API_KEY=/DEEPGRAM_API_KEY=<your-key>/' .env
   ```

2. Install dependencies using `uv`:
   ```
   make setup
   ```

3. We provide one data file in the `data` folder. Add your own audio files (`.wav`) to the data folder if you like:
   ```
   cp <your_audio.wav> data/
   ```

## Running 

To investigate audio optimization we include one audio file from [freesound.org]: [ExcessiveExposure.wav by acclivity -- https://freesound.org/s/33711/ -- License: Attribution NonCommercial 4.0](https://freesound.org/people/acclivity/sounds/33711/).

We provide scripts for transcription, visualization and optimization of the audio. For best results, run them in order.

To run all three on the given example file, you can use:
```
make run
```

Each script will create files in the `analysis` folder, using the `.wav` filename as subfolder.

1. Transcribe audio, using either the Deepgram API or local Whisper
   ```
   # deepgram
   uv run scripts/1_transcribe_audio.py --model_id nova-3 --wav_file <wav_file>

   # whisper
   uv run scripts/1_transcribe_audio.py --model_id mlx_community/whisper-small-mlx --wav_file <wav_file>
   ```

2. Visualize spectrograms and the resulting transcriptions:
   ```
   uv run scripts/2_visualize_audio.py --wav_file <wav_file>
   ```

3. Optimize audio to confuse a specific Whisper model:
   ```
   uv run scripts/3_optimize_audio.py --model_id mlx_community/whisper-small-mlx --wav_file <wav_file>
   ```
