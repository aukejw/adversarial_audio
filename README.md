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

3. Add your audio files (`.wav`) to the data folder:
   ```
   cp <your_audio.wav> data/
   ```

## Running 

For best results, run all scripts in order. 

Each script will create files in the `analysis` folder, using the `.wav` filename as subfolder.

1. Transcribe audio, using either the Deepgram API or local Whisper
   ```
   # deepgram
   uv run scripts/1_transcribe_audio.py --model_id nova-3

   # whisper
   uv run scripts/1_transcribe_audio.py --model_id mlx_community/whisper-large-v3-turbo
   ```

2. Visualize spectrograms and the resulting transcriptions:
   ```
   uv run scripts/2_visualize_audio.py
   ```

3. Optimize audio based on the Whisper model:
   ```
   uv run scripts/4_optimize_audio.py
   ```

4. Optionally, we can use the transcriptions for voice cloning too:
   ```
   uv run scripts/3_generate_audio.py --tts_model_id mlx_community/Dia-1.6B
   ```
