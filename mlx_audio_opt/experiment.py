from pathlib import Path
from typing import Optional, Union

import librosa
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn
import numpy as np
import soundfile
from mlx_whisper.audio import (
    HOP_LENGTH,
    N_FFT,
    N_FRAMES,
    SAMPLE_RATE,
    hanning,
    pad_or_trim,
)
from mlx_whisper.transcribe import ModelHolder, get_tokenizer
from tqdm import tqdm

from mlx_audio_opt.audio.spectrogram import Spectrogram
from mlx_audio_opt.audio.stft import magnitudes_to_log_mel_spectrogram
from mlx_audio_opt.stt import whisper
from mlx_audio_opt.visualization.plot_audio import compare_spectrograms


class AdversarialAudioExperiment:
    """Adversarial audio optimization experiment.

    Optimizes a single .wav file to maximally confuse the Whisper model.

    """

    def __init__(
        self,
        wav_file: Union[str, Path],
        model_id: str,
        transcription_file: str,
        output_folder: Union[str, Path],
    ):
        self.wav_file = Path(wav_file)
        self.model_id = model_id
        self.transcription_file = Path(transcription_file)

        assert self.wav_file.exists(), f"wav_file '{wav_file}' does not exist."
        assert (
            self.transcription_file.exists()
        ), f"transcription_file '{transcription_file}' must exist."

        self.short_model_id = model_id.split("/")[-1]
        self.output_folder = Path(output_folder) / self.short_model_id
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.setup_model()
        self.load_audio()
        self.load_transcription()

    def setup_model(self):
        """Load the model."""
        self.dtype = mx.float16
        self.model = ModelHolder.get_model(self.model_id, self.dtype)
        self.tokenizer = get_tokenizer(
            self.model.is_multilingual,
            num_languages=self.model.num_languages,
            language="en",
            task="transcribe",
        )

    def load_audio(self):
        """Load the audio file."""

        # Get the spectrogram using librosa, we'll use the phase for audio reconstruction.
        # This is not a perfect match with mlx_whisper's stft, but the result is close.
        audio_series, sampling_rate = librosa.load(str(self.wav_file), sr=SAMPLE_RATE)
        self.sampling_rate = sampling_rate

        # these stft kwargs should be used during the inverse stft as well!
        self.stft_kwargs = dict(
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=N_FFT,
            center=False,
            window=np.array(hanning(N_FFT)),
        )
        complex_spectrogram = librosa.stft(
            audio_series,
            **self.stft_kwargs,
            pad_mode="reflect",
        )

        # we will optimize the magnitude (power 1) to avoid large numbers
        magnitudes = np.abs(complex_spectrogram)
        phase = np.angle(complex_spectrogram)
        self.original_spectrogram = Spectrogram(magnitudes, phase)

        # Initialize the tensor to optimize.
        # mlx_whisper works with <mel> x <n_frames> arrays, padded to N_FRAMES frames
        self.magnitudes_tensor = pad_or_trim(
            self.original_spectrogram.whisper_tensor,
            length=N_FRAMES,
            axis=-2,
        ).astype(self.dtype)

    def load_transcription(self):
        """Load the given transcription from file."""
        original_tokens = whisper.get_tokens_from_transcription(
            tokenizer=self.tokenizer,
            transcription=self.transcription_file,
            add_eot_token=True,
        )
        sequence_str = self.tokenizer.decode_with_timestamps(original_tokens)
        print(f"Original transcription:\n{sequence_str}")

        target_tokens = (
            list(self.tokenizer.sot_sequence)
            + self.tokenizer.encode(
                " Here's a short paragraph about the nature of spectrograms. "
                "Spectrograms are a visual representation of the spectrum of frequencies "
                "in a sound or other signal as they vary with time. "
            )
            + [self.tokenizer.eot]
        )
        sequence_str = self.tokenizer.decode_with_timestamps(target_tokens)
        print(f"Original transcription:\n{sequence_str}")

        original_tokens: mx.array = mx.array(original_tokens)
        original_tokens = mx.broadcast_to(original_tokens, (1, len(original_tokens)))
        self.original_tokens = original_tokens

        # Initialize token array
        target_tokens: mx.array = mx.array(target_tokens)
        target_tokens = mx.broadcast_to(target_tokens, (1, len(target_tokens)))
        self.target_tokens = target_tokens

        # Mask loss for tokens that are the same as the original tokens
        num_target_tokens = target_tokens.shape[1]
        self.target_token_mask = mx.ones((num_target_tokens,), dtype=self.dtype)
        for index in range(num_target_tokens):
            if target_tokens[0, index] == original_tokens[0, index]:
                self.target_token_mask[index] = 0.0

    def run(
        self,
        num_iterations: int = 1_000,
        log_every_n: int = 100,
        learning_rate: float = 1e-1,
        max_grad: float = 5.0,
    ):
        """Run the experiment."""

        target_tokens = self.target_tokens
        magnitudes_tensor = self.magnitudes_tensor

        print(f"Performing optimization:")
        print(f"  Magnitudes shape:    {magnitudes_tensor.shape}")
        print(f"  Target tokens shape: {target_tokens.shape}")

        progressbar = tqdm(range(num_iterations), desc="Optimizing audio")
        iteration = 0

        try:
            for iteration in progressbar:
                # Compute gradient wrt the input magnitudes
                loss_and_grad_fn = mx.value_and_grad(
                    self.get_nll,
                    argnums=0,  # optimize wrt the first argument passed to the function
                )
                nll, grads = loss_and_grad_fn(
                    magnitudes_tensor,
                    self.model,
                    target_tokens,
                    self.target_token_mask,
                )

                # padding frames get no gradients -- here quite straightforward
                grads[self.original_spectrogram.num_frames :] = 0.0

                # clip to avoid explosion
                grads = mx.clip(grads, -max_grad, max_grad)

                # perform simple gradient descent, maximizing log p(new_tokens|audio)
                magnitudes_tensor = magnitudes_tensor - learning_rate * grads
                mx.eval(magnitudes_tensor)

                progressbar.set_description(f"Optimizing audio, nll = {nll:.4f}")

                # Log intermediate results every so often
                if iteration % log_every_n == 0 or iteration == num_iterations - 1:
                    self.log_intermediate_results(
                        iteration=iteration,
                        nll=nll,
                        grads=grads,
                        magnitudes_tensor=magnitudes_tensor,
                    )
        except KeyboardInterrupt:
            print("Interrupted by user..")

        print(f"Optimization finished at iteration {iteration}.")
        print(f"  Final audio saved to '{self.output_wav_file}'")
        print(f"  Final neg log likelihood = '{nll:.4f}'")
        return self.output_wav_file

    def get_log_probs(
        self,
        magnitudes: mx.array,
        model: mlx.nn.Module,
        tokens: mx.array,
    ) -> mx.array:
        """Perform inference. Return neg log likelihood of the token sequence."""
        # convert magnitude to the log mel spectrogram Whisper expects
        mel = magnitudes_to_log_mel_spectrogram(
            magnitudes**2,
            n_mels=model.dims.n_mels,
        )

        # encode audio, decode logits
        mel = mel[None]
        audio_features = model.encoder(mel)
        logits, kv_cache, _ = model.decoder(tokens, audio_features)
        log_probs = mlx.nn.log_softmax(logits, axis=-1)

        # Quick sanity check
        log_probs = mx.squeeze(log_probs)
        num_tokens = tokens.shape[1]
        vocab_size = self.model.dims.n_vocab
        assert log_probs.shape == (num_tokens, vocab_size), log_probs.shape

        # select the log probabilities of the token sequence, p(tokens|audio)
        log_probs = log_probs[mx.arange(num_tokens), tokens[0]]
        assert log_probs.shape == (num_tokens,), log_probs.shape

        return log_probs

    def get_nll(
        self,
        magnitudes: mx.array,
        model: mlx.nn.Module,
        tokens: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Perform inference. Return neg log likelihood of the token sequence.

        Args:
            magnitudes: The input spectrograms magnitudes (power=1).
            model: The Whisper model instance.
            tokens: The token sequence to compute the nll for.
            token_mask: If given, masks the nlls before computing the loss.

        """
        target_log_probs = self.get_log_probs(
            magnitudes=magnitudes,
            model=model,
            tokens=self.target_tokens,
        )
        # Subselect log probs if needed
        if mask is not None:
            target_log_probs = mask * target_log_probs

        # return the neg log (joint) likelihood of the last token
        return -mx.sum(target_log_probs)

    def log_intermediate_results(
        self,
        iteration: int,
        nll: mx.array,
        grads: mx.array,
        magnitudes_tensor: mx.array,
    ):
        """Log results to console and to file."""
        mt = magnitudes_tensor

        print(f"  Iteration {iteration+1}")
        print(f"    neg log likelihood  = {nll:.4f}")
        print(f"    grads min, max      = {grads.min():+.2f}, {grads.max():+.2f}")
        print(f"    magnitudes min, max = {mt.min():+.2f}, {mt.max():+.2f}")

        self.print_log_probs(
            magnitudes_tensor=magnitudes_tensor,
        )

        # Show visuals
        spectrogram = Spectrogram.from_whisper(
            magnitudes_tensor[: self.original_spectrogram.num_frames, :],
        )

        self.show_intermediate_results(
            magnitudes=spectrogram.magnitudes,
            original_magnitudes=self.original_spectrogram.magnitudes,
            iteration=iteration,
        )
        output_wav_file = self.save_audio(
            magnitudes=spectrogram.magnitudes,
            phase=self.original_spectrogram.phase,
            iteration=iteration,
        )
        self.output_wav_file = output_wav_file
        return output_wav_file

    def print_log_probs(
        self,
        magnitudes_tensor: mx.array,
    ):
        """Print log probabilities of the original and target tokens."""
        log_probs_original = self.get_log_probs(
            magnitudes=magnitudes_tensor,
            model=self.model,
            tokens=self.original_tokens,
        )
        nll_original = -mx.sum(log_probs_original)

        log_probs_target = self.get_log_probs(
            magnitudes=magnitudes_tensor,
            model=self.model,
            tokens=self.target_tokens,
        )
        nll_target = -mx.sum(log_probs_target)

        def print_original_and_target(
            token1: int,
            token2: int,
            log_prob1: float,
            log_prob2: float,
        ):
            if token1 is not None:
                print(f"{token1:>25} log p={log_prob1:>9.2f} ", end=" " * 8)
            else:
                print(f"{'    ':>25}       {'          ':>9} ", end=" " * 8)

            if token2 is not None:
                print(f"{token2:>25} p={log_prob2:>9.2f}")
            else:
                print("")

        print("\nLog probabilities of original and target sentence:")
        print_original_and_target(
            token1="original",
            token2="target",
            log_prob1=-nll_original,
            log_prob2=-nll_target,
        )
        for token_index in range(
            max(len(self.original_tokens[0]), len(self.target_tokens[0]))
        ):
            original_token = target_token = None

            if token_index < len(log_probs_original):
                original_token = int(self.original_tokens[0, token_index])
                original_token = self.tokenizer.decode_with_timestamps([original_token])
                original_log_prob = float(log_probs_original[token_index])

            if token_index < len(log_probs_target):
                target_token = int(self.target_tokens[0, token_index])
                target_token = self.tokenizer.decode_with_timestamps([target_token])
                target_log_prob = float(log_probs_target[token_index])

            print_original_and_target(
                token1=original_token,
                token2=target_token,
                log_prob1=original_log_prob,
                log_prob2=target_log_prob,
            )
        return

    def show_intermediate_results(
        self,
        magnitudes: np.ndarray,
        original_magnitudes: np.ndarray,
        iteration: int,
    ) -> Path:
        """Save intermediate results to file."""
        n_mels = self.model.dims.n_mels

        # Compare absolute magnitudes
        output_png_file = (
            self.output_folder / f"optimized_audio_it{iteration+1:0>6}_magnitude.png"
        )
        compare_spectrograms(
            magnitudes1=magnitudes,
            magnitudes2=original_magnitudes,
            title1="Optimized",
            title2="Original",
            suptitle="Optimized vs original",
        )
        plt.savefig(output_png_file, bbox_inches="tight", dpi=300)
        plt.close()

        # Compare log-Mel spectrograms
        output_png_file = (
            self.output_folder / f"optimized_audio_it{iteration+1:0>6}_mel.png"
        )
        mel1 = magnitudes_to_log_mel_spectrogram(
            mx.array(magnitudes).T ** 2,
            n_mels=n_mels,
        ).T
        mel2 = magnitudes_to_log_mel_spectrogram(
            mx.array(original_magnitudes).T ** 2,
            n_mels=n_mels,
        ).T
        compare_spectrograms(
            magnitudes1=mel1,
            magnitudes2=mel2,
            title1="Optimized",
            title2="Original",
            suptitle="Optimized vs original",
        )
        plt.savefig(output_png_file, bbox_inches="tight", dpi=300)
        plt.close()

        return output_png_file

    def save_audio(
        self,
        magnitudes: np.ndarray,
        phase: np.ndarray,
        iteration: int,
    ) -> Path:
        """Reconstruct audio, save to file."""

        complex_spectrogram = magnitudes * np.exp(1j * phase)

        audio_series = librosa.istft(
            complex_spectrogram,
            **self.stft_kwargs,
        )
        output_wav_file = (
            self.output_folder / f"optimized_audio_it{iteration+1:0>6}.wav"
        )
        soundfile.write(
            output_wav_file.as_posix(),
            audio_series,
            samplerate=self.sampling_rate,
        )
        return output_wav_file
