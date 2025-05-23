"""

Compare two transcriptions. 
 
For example, run on the example data file:

expdir=analysis/33711__acclivity__excessiveexposure/4_modify_audio/exp_2025-05-22_11-46-09
uv run python scripts/compare_transcriptions.py \
    --wav_file1 data/33711__acclivity__excessiveexposure.wav \
    --wav_file2 $expdir/iteration_000500/optimized_audio.wav \
    --transcription1 $expdir/transcription_before.json \
    --transcription2 $expdir/transcription_after.json \
    --output_dir $expdir/comparison_video

"""

import shutil
from pathlib import Path
from typing import List

import fire
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
from moviepy import AudioFileClip, ImageSequenceClip
from tqdm import tqdm

from mlx_audio_opt.stt.transcription import get_transcription
from mlx_audio_opt.visualization.plot_transcription import plot_transcription_words

plt.style.use("dark_background")


def main(
    wav_file1: str,
    wav_file2: str,
    transcription1: str,
    transcription2: str,
    output_dir: str,
    padding_seconds: float = 0.1,
    dont_delete_frames: bool = False,
):
    """Create a video comparing two transcriptions.

    Args:
        wav_file1 (str): Path to the first wav file.
        wav_file2 (str): Path to the second wav file.
        transcription1 (str): Path to the first transcription file.
        transcription2 (str): Path to the second transcription file.
        output_dir (str): Directory to save the output video.
        padding_seconds (float): We add a few seconds of silence between audio snippets.
        dont_delete_frames (bool): If true, do not remove frame pngs after video creation.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_file1 = Path(wav_file1)
    wav_file2 = Path(wav_file2)
    transcription1 = get_transcription(transcription1)
    transcription2 = get_transcription(transcription2)
    words1 = transcription1.get_words()
    words2 = transcription2.get_words()

    # Create stitched audio
    sampling_rate, audio_duration, audio_file = create_audio(
        wav_file1=wav_file1,
        wav_file2=wav_file2,
        output_dir=output_dir,
        padding_seconds=padding_seconds,
    )

    # Render frames
    frame_folder = output_dir / "frames"
    frame_folder.mkdir(parents=True, exist_ok=True)
    frame_paths, durations = render_frames(
        frame_folder=frame_folder,
        padding_seconds=padding_seconds,
        words1=words1,
        words2=words2,
        audio_duration=audio_duration,
    )

    # Create the clip
    clip = ImageSequenceClip(
        sequence=[path.as_posix() for path in frame_paths],
        durations=durations,
    )
    audio_clip = AudioFileClip(audio_file.as_posix())
    clip.audio = audio_clip

    video_file = output_dir / "transcription_comparison.mp4"
    clip.write_videofile(
        video_file.as_posix(),
        codec="libx264",
        audio_codec="aac",
        audio=True,
        fps=24,
        audio_fps=sampling_rate,
        preset="fast",
        ffmpeg_params=[
            "-pix_fmt",
            "yuv420p",
            "-vf",
            f"scale={1280}:-2",
        ],
    )

    if dont_delete_frames:
        print(f"Deleting framefolder '{frame_folder}'")
        shutil.rmtree(frame_folder)

    print(f"Done! Your comparison video is saved at '{video_file}'")


def create_audio(
    wav_file1: Path,
    wav_file2: Path,
    output_dir: Path,
    padding_seconds: float,
):
    """Create a stitched audio file from two wav files."""
    audio1, sampling_rate = librosa.load(wav_file1.as_posix(), sr=None)
    audio2, sampling_rate = librosa.load(wav_file2.as_posix(), sr=sampling_rate)
    audio_duration = librosa.get_duration(y=audio1, sr=sampling_rate)

    padding_audio = np.zeros(int(sampling_rate * padding_seconds))
    audio = np.concatenate(
        [
            padding_audio,
            audio1,
            padding_audio,
            audio2,
            padding_audio,
        ],
        axis=0,
    )
    audio_file = output_dir / "combined_audio.wav"
    soundfile.write(
        file=audio_file.as_posix(),
        data=audio,
        samplerate=sampling_rate,
    )

    return sampling_rate, audio_duration, audio_file


def render_frames(
    frame_folder: Path,
    padding_seconds: float,
    words1: List,
    words2: List,
    audio_duration: float,
):
    """Render frames as png files. Return their paths and durations."""
    frame_paths = []
    durations = []

    word_indices = [(i, 0) for i in range(1, len(words1) + 1)]
    word_indices += [(len(words1), i) for i in range(1, len(words2) + 1)]

    words_start = min(words1[0]["start"], words2[0]["start"])
    words_end = max(words1[-1]["end"], words2[-1]["end"])

    # first frame
    frame_path = frame_folder / "frame_0000.png"
    plot_transcription_frame(
        words1_so_far=[],
        words2_so_far=[],
        frame_path=frame_path,
        xlim=[words_start - 0.1, words_end + 0.1],
    )
    frame_paths.append(frame_path)
    durations.append(padding_seconds)

    iterator = tqdm(
        enumerate(word_indices, start=1),
        total=len(word_indices),
        unit="frame",
    )

    for frame_index, (word1_index, word2_index) in iterator:
        iterator.set_description(
            f"Rendering frames, word1={word1_index}, word2={word2_index}"
        )
        words1_so_far = words1[:word1_index]
        words2_so_far = words2[:word2_index] if word2_index > 0 else []

        # plot frame, save to file
        frame_path = frame_folder / f"frame_{frame_index:0>4}.png"
        plot_transcription_frame(
            words1_so_far=words1_so_far,
            words2_so_far=words2_so_far,
            frame_path=frame_path,
            xlim=[words_start - 0.1, words_end + 0.1],
        )

        # determine duration
        last_word = words2_so_far[-1] if len(words2_so_far) else words1_so_far[-1]
        duration = last_word["end"] - last_word["start"]

        # special cases: we need to let first and last words hang for a bit
        if word1_index == len(words1) and word2_index == 0:
            duration += (audio_duration - last_word["end"]) + padding_seconds
        if word2_index == len(words2):
            duration += (audio_duration - last_word["end"]) + padding_seconds

        frame_paths.append(frame_path)
        durations.append(duration)

    return frame_paths, durations


def _format_title(
    prefix: str,
    words: List[str],
) -> str:
    """Create a plot title."""
    if len(words) == 0:
        return "\n"

    title = prefix + ":"
    for word_index, word in enumerate(words):
        if word_index > 0 and word_index % 13 == 0:
            title += "\n"
        else:
            title += " "
        title += word["word"].strip()

    if "\n" not in title:
        title += "\n"

    return title


def plot_transcription_frame(
    words1_so_far: List,
    words2_so_far: List,
    frame_path: Path,
    xlim: List[float] = None,
):
    """Plot transcriptions."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    title = _format_title("Original", words1_so_far)
    plot_transcription_words(
        ax=axes[0],
        title=title,
        title_fontsize=15,
        words=words1_so_far,
        xlim=xlim,
    )
    title = _format_title("Modified", words2_so_far)
    plot_transcription_words(
        ax=axes[1],
        title=title,
        title_fontsize=15,
        words=words2_so_far,
        xlim=xlim,
    )
    fig.tight_layout()
    plt.savefig(frame_path, bbox_inches="tight", dpi=300, pad_inches=0.4)
    plt.close()
    return frame_path


if __name__ == "__main__":
    fire.Fire(main)
