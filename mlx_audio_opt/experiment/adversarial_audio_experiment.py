import copy
from pathlib import Path
from typing import Any, Dict, List, Union

import librosa
import mlx.core as mx
import numpy as np
import soundfile
from mlx_whisper.audio import N_FRAMES, SAMPLE_RATE
from mlx_whisper.transcribe import ModelHolder, get_tokenizer
from tqdm import tqdm

from mlx_audio_opt.audio.istft import reconstruct_audio_from_spectrogram
from mlx_audio_opt.audio.spectrogram import Spectrogram
from mlx_audio_opt.audio.stft import get_spectrogram, magnitudes_to_log_mel_spectrogram
from mlx_audio_opt.file_io import to_json
from mlx_audio_opt.stt import whisper
from mlx_audio_opt.stt.transcription import WhisperTranscription
from mlx_audio_opt.visualization.plot_audio import compare_spectrograms, save_and_close
from mlx_audio_opt.visualization.print_tokens import print_sentence


class AdversarialAudioExperiment:
    """Adversarial audio optimization experiment.

    Optimizes a single .wav file to maximally confuse the Whisper model.

    """

    def __init__(
        self,
        wav_file: Union[str, Path],
        model_id: str,
        output_folder: Union[str, Path],
    ):
        self.wav_file = Path(wav_file)
        self.model_id = model_id

        assert self.wav_file.exists(), f"wav_file '{wav_file}' does not exist."

        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.setup_model()
        self.setup_spectrogram()

    def setup_model(self):
        """Load the model."""
        assert "mlx" in self.model_id and "whisper" in self.model_id, (
            "The stft used here is implemented only for mlx-whisper. If a "
            "different model is used, you'll need to adapt the stft and "
            "istft functions accordingly. "
        )
        self.dtype = mx.float16
        self.model = ModelHolder.get_model(self.model_id, self.dtype)
        self.tokenizer = get_tokenizer(
            self.model.is_multilingual,
            num_languages=self.model.num_languages,
            language="en",
            task="transcribe",
        )

    def setup_spectrogram(self):
        """Load the audio file and store the (unpadded) spectrogram."""

        # Load audio, compute (padded) spectrogram
        self.audio_series, self.sampling_rate = librosa.load(
            self.wav_file.as_posix(),
            sr=SAMPLE_RATE,
        )
        self.spectrogram = get_spectrogram(self.audio_series)

        # Audio was padded by N_SAMPLES before STFT, we remove this here.
        # N_SAMPLES waveform should result in N_FRAMES frames worth of padding.
        # We can now use this unpadded spectrogram to reconstruct audio later.
        self.n_content_frames = self.spectrogram.num_frames - N_FRAMES
        self.unpadded_spectrogram = self.spectrogram.trim(self.n_content_frames)

    def transcribe(
        self,
        audio_file: Union[str, Path],
        json_file_name: Union[str, Path] = None,
    ) -> Dict[str, Any]:
        """Transcribe the given audio file."""
        mx.random.seed(0)

        transcription: WhisperTranscription = whisper.transcribe_audio(
            audio=audio_file,
            model_id=self.model_id,
            fp16=True if self.dtype == mx.float16 else False,
        )
        if json_file_name is not None:
            to_json(
                transcription.transcription,
                self.output_folder / json_file_name,
            )

        tokens = (
            list(self.tokenizer.sot_sequence)  # start of sentence, language, task
            + transcription.get_tokens()  # our transcription
            + [self.tokenizer.eot]  # end of sentence
        )
        sequence_str = self.tokenizer.decode_with_timestamps(tokens)

        return dict(
            transcription=transcription,
            tokens=tokens,
            sequence_str=sequence_str,
        )

    def run(
        self,
        num_iterations: int = 1_000,
        log_every_n: int = 100,
        reload_audio_every_n: int = 20,
        learning_rate: float = 1e-1,
        l2_penalty: float = 0.0,
    ):
        """Run the experiment."""
        mx.random.seed(0)

        transcription_dict = self.transcribe(
            self.wav_file,
            json_file_name="transcription_before.json",
        )
        self.tokens: List[int] = transcription_dict["tokens"]
        self.transcription: WhisperTranscription = transcription_dict["transcription"]

        # Initialize the tensor to optimize (same shape as padded magnitudes tensor)
        magnitudes_tensor = self.spectrogram.whisper_tensor
        self.original_magnitudes_tensor = copy.deepcopy(magnitudes_tensor)

        # Initialize the token tensor.
        token_tensor: mx.array = mx.array(self.tokens)
        token_tensor = mx.broadcast_to(token_tensor, (1, len(token_tensor)))

        print(f"Performing optimization:")
        print(f"  Learning rate:    {learning_rate}")
        print(f"  L2 penalty:       {l2_penalty}")
        print(f"  Num iterations:   {num_iterations}")
        print(f"  Magnitudes shape: {magnitudes_tensor.shape}")
        print(f"  Tokens shape:     {token_tensor.shape}")

        nll = self.get_nll(
            magnitudes=magnitudes_tensor,
            tokens=token_tensor,
        )
        print(f"  Original nll: {nll:.4f}")
        print(f"  Transcription before:\n  {transcription_dict['sequence_str']}")

        progressbar = tqdm(range(num_iterations), desc="Optimizing audio")
        iteration = 0

        for iteration in progressbar:
            # Compute gradient wrt the input
            loss_and_grad_fn = mx.value_and_grad(
                self.get_nll,
                argnames="magnitudes",  # optimize loss wrt input
            )
            loss, grads = loss_and_grad_fn(
                magnitudes=magnitudes_tensor,
                tokens=token_tensor,
                l2_penalty=l2_penalty,
            )
            grads = grads[1]["magnitudes"]

            # don't alter padding regions, they will be cut off later
            grads[:, self.n_content_frames :] = 0

            # gradient ascent, minimizing log p(tokens|magnitudes)
            magnitudes_tensor = magnitudes_tensor + learning_rate * grads
            mx.eval(magnitudes_tensor)

            progressbar.set_description(f"Optimizing audio, loss = {loss:.4f}")

            # Log intermediate results every so often
            if iteration % log_every_n == 0 or iteration == num_iterations - 1:
                self.log_intermediate_results(
                    iteration=iteration,
                    loss=loss,
                    grads=grads,
                    magnitudes_tensor=magnitudes_tensor,
                    tokens_tensor=token_tensor,
                )
            # Reload audio (move to waveform and back) every so often
            if iteration % reload_audio_every_n == 0:
                magnitudes_tensor = self.reload_audio(
                    magnitudes_tensor=magnitudes_tensor,
                )

        print(f"\nOptimization finished after {iteration+1} iterations.")
        print(f"  Final audio saved to '{self.optimized_wav_file}'")

        nll = self.get_nll(
            magnitudes=magnitudes_tensor,
            tokens=token_tensor,
        )
        transcription_dict = self.transcribe(
            self.optimized_wav_file,
            json_file_name="transcription_after.json",
        )
        print(f"  Final nll: {nll:.4f}")
        print(f"  Transcription after:\n  {transcription_dict['sequence_str']}")

        return self.optimized_wav_file

    def reload_audio(
        self,
        magnitudes_tensor: mx.array,
    ) -> mx.array:
        """Reload the audio by going back to waveform and re-computing the STFT.

        This is necessary because appling a istft to an altered spectrorgram is
        not a lossless operation. Minor changes in the spectrogram may be removed
        or altered, and the resulting transcription will be very different from
        the one we saw during optimization.

        To avoid this, we convert the altered spectrogram to audio, then apply
        a stft to the new audio to get a spectrogram again. This new spectrogram
        is much closer to the one we'd see if we load audio from file.

        """
        magnitudes = magnitudes_tensor[0 : self.n_content_frames].T

        # create the current spectrogram (not padded)
        unpadded_spectrogram = Spectrogram(
            magnitudes=magnitudes,
            phase=self.unpadded_spectrogram.phase,
        )

        # move spectrogram -> audio -> spectrogram
        audio_series = reconstruct_audio_from_spectrogram(
            spectrogram=unpadded_spectrogram,
        )
        spectrogram = get_spectrogram(
            audio_series,
            remove_last_frame=False,
        )

        # use the resulting magnitudes as our new model input
        magnitudes_tensor = spectrogram.whisper_tensor
        return magnitudes_tensor

    def get_nll(
        self,
        magnitudes: mx.array,
        tokens: mx.array,
        l2_penalty: float = 0.0,
    ) -> mx.array:
        """Perform inference. Return neg log likelihood of the token sequence.

        Args:
            magnitudes: The input spectrograms magnitudes (power=1).
            tokens: The token sequence to compute the nll for.
            l2_penalty: The L2 penalty to apply to the difference.

        Returns:
            mx.array: nll of the token sequence.

        """
        target_log_probs = whisper.get_log_probabilities(
            magnitudes=magnitudes,
            model=self.model,
            tokens=tokens,
        )

        # Avoid changing SOT, language, task, timestamp and timestamp, EOT tokens
        target_log_probs[:4] = mx.stop_gradient(target_log_probs[:4])
        target_log_probs[-2:] = mx.stop_gradient(target_log_probs[-2:])

        nll = -mx.sum(target_log_probs)

        # Compute L2 penalty if needed
        l2_loss = 0
        if l2_penalty > 0:
            residuals = magnitudes - self.original_magnitudes_tensor
            l2_loss = l2_penalty * (0.5 * mx.sum(mx.square(residuals)))

        # return the neg log (joint) likelihood of the last token
        return nll + l2_loss

    def log_intermediate_results(
        self,
        iteration: int,
        loss: mx.array,
        grads: mx.array,
        magnitudes_tensor: mx.array,
        tokens_tensor: mx.array,
    ):
        """Log results to console and to file."""
        mt = magnitudes_tensor

        print(f"")
        print(f"  Iteration {iteration+1}")
        print(f"    loss                = {loss:.4f}")
        print(f"    grads min, max      = {grads.min():+.2f}, {grads.max():+.2f}")
        print(f"    magnitudes min, max = {mt.min():+.2f}, {mt.max():+.2f}")
        print(f"")

        # Print individual probabilities
        log_probs = whisper.get_log_probabilities(
            magnitudes=magnitudes_tensor,
            tokens=tokens_tensor,
            model=self.model,
        )
        print_sentence(
            tokens=self.tokens,
            log_probs=log_probs,
            tokenizer=self.tokenizer,
        )

        # Show visuals
        output_folder = self.output_folder / f"iteration_{iteration+1:0>6}"
        output_folder.mkdir(parents=True, exist_ok=True)

        spectrogram = Spectrogram(
            magnitudes=magnitudes_tensor[0 : self.n_content_frames].T,
            phase=self.unpadded_spectrogram.phase,
        )

        self.show_intermediate_results(
            magnitudes=spectrogram.magnitudes,
            original_magnitudes=self.unpadded_spectrogram.magnitudes,
            output_folder=output_folder,
        )
        optimized_wav_file = self.save_audio(
            spectrogram=spectrogram,
            output_folder=output_folder,
        )
        self.optimized_wav_file = optimized_wav_file
        return optimized_wav_file

    def show_intermediate_results(
        self,
        magnitudes: np.ndarray,
        original_magnitudes: np.ndarray,
        output_folder: Path,
    ) -> Path:
        """Save intermediate results to file."""
        n_mels = self.model.dims.n_mels

        # Compare absolute magnitudes
        fig = compare_spectrograms(
            magnitudes1=magnitudes,
            magnitudes2=original_magnitudes,
            title1="Optimized",
            title2="Original",
            suptitle="Optimized vs original",
            ylabel="frequency bins",
        )
        output_png_file = output_folder / f"optimized_audio_magnitude.png"
        save_and_close(fig, output_png_file)

        # Compare log-Mel spectrograms
        mel1 = magnitudes_to_log_mel_spectrogram(
            mx.array(magnitudes).T,
            n_mels=n_mels,
        ).T
        mel2 = magnitudes_to_log_mel_spectrogram(
            mx.array(original_magnitudes).T,
            n_mels=n_mels,
        ).T
        fig = compare_spectrograms(
            magnitudes1=mel1,
            magnitudes2=mel2,
            title1="Optimized",
            title2="Original",
            suptitle="Optimized vs original",
            ylabel="log mel frequency bins",
        )
        output_png_file = output_folder / f"optimized_audio_log_mel.png"
        save_and_close(fig, output_png_file)

        return output_png_file

    def save_audio(
        self,
        spectrogram: Spectrogram,
        output_folder: Path,
    ) -> Path:
        """Reconstruct audio, save to file."""
        audio_series = reconstruct_audio_from_spectrogram(
            spectrogram=spectrogram,
            length=self.audio_series.shape[0],
        )

        optimized_wav_file = output_folder / f"optimized_audio.wav"
        soundfile.write(
            optimized_wav_file.as_posix(),
            audio_series,
            samplerate=self.sampling_rate,
        )
        transcription_dict = self.transcribe(audio_file=optimized_wav_file)

        print(f"  Transcription currently:\n  {transcription_dict['sequence_str']}\n")

        return optimized_wav_file
