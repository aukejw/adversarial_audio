# MLX audio optimization

Using pretrained models to investigate audio data on Apple silicon.

Our goal is to find adversarial examples for Whisper models uing `mlx_whisper`. 
We achieve this by computing the gradient of the nll with respect to inputs, and performing stochastic gradient ascent:

```
   log p(T | s, x) = \sum f(T_i | T_{<i}, x, θ)
                 x = x + ∇x log p(T|s, x, θ)
```

Iteratively updating the input audio can fool Whisper models into producing the wrong output. Example applications of these adversarial examples include:

  - making speech models more robust by incorporating them during training
  - stresstesting and finding commonly confused tokens
  - pentesting applications that rely on speech input

## What to expect

This library enables you to fool open-weight ASR models. We include one audio file from freesound.org as an example: [ExcessiveExposure.wav by acclivity](https://freesound.org/people/acclivity/sounds/33711/) (License: Attribution NonCommercial 4.0).

Before optimization, this file is transcribed accurately by `mlx-community/mlx-whisper-small` as:

```
We will not be held responsible for any hearing impairments or
damage caused to you from excessive exposure to this sound.
```

After just 100 iterations of adversarial optimization, this Whisper model is thoroughly confused, transcribing text as:

```
We will not be held responsible for any peering impendence or
damage goes to yield on excessive exclserty list sand.
```

The transcription probabilities are shown below:

![afbeelding](https://github.com/user-attachments/assets/a90e36d4-be69-4d4a-98b2-cdc469ff2844)


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

## Disclaimer

This project enables creating adversarial examples intended for research and educational use. You may not:

  - use it to cause harm
  - use it illegally obtain access to systems

By using this project, you agree to uphold legal and ethical standards. 
