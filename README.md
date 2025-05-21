# MLX audio optimization

Using pretrained models to investigate audio data on Apple silicon.

Our goal is to find adversarial examples for Whisper models uing `mlx_whisper`. 
We achieve this by computing the gradient of the negative log likelihood of the sentence `T` with respect to audio inputs `x`, and performing stochastic gradient ascent:

```
   log p(T | s, x) = \sum f(T_i | T_{<i}, x, θ)
                 x = x + α ∇x [ -log p(T|s, x, θ) ]
```

Iteratively updating the input audio will cause Whisper models to produce the wrong transcriptions. Armed with these adversarial examples, we can:

  - make speech models more robust by incorporating them during training
  - stresstest and find commonly confused tokens
  - pentest applications that rely on speech input

## What to expect

This library enables you to fool open-weight ASR models by modifying the input audio. We include one audio file from freesound.org as an example: [ExcessiveExposure.wav by acclivity](https://freesound.org/people/acclivity/sounds/33711/) (License: Attribution NonCommercial 4.0).

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

You can usually hear a difference - these are not imperceptible attacks. Nevertheless, a human would not transcribe the audio this way.

<details> <summary>Original audio</summary>

https://github.com/user-attachments/assets/c9a7ba82-c08f-4cb2-a2f0-63dd927f09ff

</details> 

<details><summary>Optimized audio</summary>   

https://github.com/user-attachments/assets/df9f1d3c-d1d0-4fe7-b8de-e14c6392c2d8

</details>

Both sets of transcription probabilities are shown below:

![afbeelding](https://github.com/user-attachments/assets/a90e36d4-be69-4d4a-98b2-cdc469ff2844)


## Installation

You will need:

  - Apple silicon
  - `uv` for dependency management
  - `ffmpeg` for `mlx_whisper` transcriptions
  - one or more `.wav` files
  - Optionally: a Deepgram API key (add yours to a `.env` file) for Deepgram transcriptions
  
1. Install dependencies using `uv`:
   ```
   make setup
   ```

That's all. Optionally, you can:

1. Add your own audio files (`.wav`) to the data folder:
   ```
   cp <your_audio.wav> data/
   ```

2. You can add your Deepgram API key to a `.env` file for transcription with a different model:
   ```
   cp .env.template .env
   sed -i 's/DEEPGRAM_API_KEY=/DEEPGRAM_API_KEY=<your-key>/' .env
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
   WAV_FILE=data/33711__acclivity__excessiveexposure.wav
   
   # whisper
   uv run scripts/1_transcribe_audio.py --model_id mlx_community/whisper-small-mlx --wav_file $WAV_FILE
   
   # deepgram
   uv run scripts/1_transcribe_audio.py --model_id nova-3 --wav_file $WAV_FILE
   ```

2. Visualize spectrograms and the resulting transcriptions:
   ```
   uv run scripts/2_visualize_audio.py --wav_file $WAV_FILE
   ```

3. Optimize audio to confuse a specific Whisper model:
   ```
   uv run scripts/3_optimize_audio.py --model_id mlx_community/whisper-small-mlx --wav_file $WAV_FILE
   ```

## Disclaimer

This project enables creating adversarial examples intended for research and educational use. You may not:

  - use it to cause harm
  - use it illegally obtain access to systems

By using this project, you agree to uphold legal and ethical standards. 
