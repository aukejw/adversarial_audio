from typing import List

import mlx.core as mx
from tqdm import tqdm

from mlx_audio_opt.experiment.adversarial_audio_experiment import (
    AdversarialAudioExperiment,
)
from mlx_audio_opt.stt import whisper
from mlx_audio_opt.stt.transcription import WhisperTranscription
from mlx_audio_opt.visualization.print_tokens import print_sentence


class ChangeSentenceExperiment(AdversarialAudioExperiment):
    """Adversarial audio optimization experiment.

    Optimizes a single .wav file to produce a different target sentence.

    """

    def __init__(
        self,
        *,
        target_sentence: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_sentence = target_sentence

    def _setup_target_sentence(
        self,
        original_tokens: List[int],
    ) -> List[int]:
        """Set up the target sentence tokens array.

        We use the original tokens to determine the first and last timestamp token.
        We ignore all other timestamp tokens in the original sentence.

        """
        tokens = (
            list(original_tokens[:4])  # sot, language, task, timestamp
            + self.tokenizer.encode(self.target_sentence)
            + original_tokens[-2:]  # timestamp, eot
        )

        sentence_str = self.tokenizer.decode_with_timestamps(tokens)
        print(f"\nTarget sentence:\n  {sentence_str}\n")

        self.target_tokens = tokens
        return tokens

    def _print_nll(
        self,
        magnitudes_tensor: mx.array,
    ):
        """Print the nll for both original and target sentences.

        Args:
            magnitudes_tensor: The (padded) magnitudes tensor.

        """
        print("\nSentence nll:")
        for name, tokens in [
            ("original", self.original_tokens),
            ("target", self.target_tokens),
        ]:
            token_tensor = mx.array(tokens)
            token_tensor = mx.broadcast_to(token_tensor, (1, len(token_tensor)))
            nll = self.get_nll(
                magnitudes=magnitudes_tensor,
                tokens=token_tensor,
            )
            print(f"  {name:<15} nll={nll:.4f}")
        print()

    def run(
        self,
        num_iterations: int = 1_000,
        log_every_n: int = 100,
        reload_audio_every_n: int = 20,
        learning_rate: float = 1e-1,
    ):
        """Run the experiment."""
        mx.random.seed(0)

        transcription_dict = self.transcribe(
            self.wav_file,
            json_file_name="transcription_before.json",
        )
        self.original_tokens: List[int] = transcription_dict["tokens"]
        self.transcription: WhisperTranscription = transcription_dict["transcription"]

        self._setup_target_sentence(original_tokens=self.original_tokens)

        # Initialize the tensor to optimize (padded magnitudes tensor)
        magnitudes_tensor = self.spectrogram.whisper_tensor

        # Initialize the token tensor.
        self.tokens = self.target_tokens
        token_tensor: mx.array = mx.array(self.tokens)
        token_tensor = mx.broadcast_to(token_tensor, (1, len(token_tensor)))

        original_token_tensor: mx.array = mx.array(self.original_tokens)
        original_token_tensor = mx.broadcast_to(
            original_token_tensor, (1, len(original_token_tensor))
        )

        print(f"Performing optimization:")
        print(f"  Learning rate:    {learning_rate}")
        print(f"  Num iterations:   {num_iterations}")
        print(f"  Magnitudes shape: {magnitudes_tensor.shape}")
        print(f"  Tokens shape:     {token_tensor.shape}")
        print(f"  Transcription before:\n  {transcription_dict['sequence_str']}\n")

        self._print_nll(magnitudes_tensor=magnitudes_tensor)

        progressbar = tqdm(range(num_iterations), desc="Optimizing audio")
        iteration = 0

        for iteration in progressbar:
            # Compute gradient of log p(target_tokens) wrt the input magnitudes
            loss_and_grad_fn = mx.value_and_grad(
                self.get_nll,
                argnames="magnitudes",  # optimize loss wrt magnitudes
            )
            nll, grads = loss_and_grad_fn(
                magnitudes=magnitudes_tensor,
                tokens=token_tensor,
            )
            grads = grads[1]["magnitudes"]

            # don't alter padding regions, they will be cut off later
            grads[:, self.n_content_frames :] = 0

            # gradient descent, maximizing log p(target_tokens|magnitudes)
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
                    tokens_tensor=token_tensor,
                )
                # Print individual probabilities
                log_probs = whisper.get_log_probabilities(
                    magnitudes=magnitudes_tensor,
                    tokens=original_token_tensor,
                    model=self.model,
                )
                print_sentence(
                    tokens=self.original_tokens,
                    log_probs=log_probs,
                    tokenizer=self.tokenizer,
                )
                self._print_nll(magnitudes_tensor=magnitudes_tensor)

            # Reload audio (move to waveform and back) every so often
            if iteration % reload_audio_every_n == 0:
                magnitudes_tensor = self.reload_audio(magnitudes_tensor)

        print(f"\nOptimization finished after {iteration+1} iterations.")
        print(f"  Final audio saved to '{self.optimized_wav_file}'")

        transcription_dict = self.transcribe(
            self.optimized_wav_file,
            json_file_name="transcription_after.json",
        )
        print(f"  Transcription after:\n  {transcription_dict['sequence_str']}\n")
        self._print_nll(magnitudes_tensor=magnitudes_tensor)

        return self.optimized_wav_file
