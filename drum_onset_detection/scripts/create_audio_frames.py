import argparse
import time
import os
import sys

import numpy as np

from tqdm.contrib.concurrent import thread_map
from pathlib import Path
from functools import partial

sys.path.append('..')
from drum_onset_detection.tools.audio_tools.in_out import init_audio, read_audio

def export_individual_frames(audio_path: Path, out_dir: Path, frame_len: int):
    """
    Load an audio file at audio_path, divide into into frames of length frame_len,
     and save each frame as an individual .npy file.

    :param audio_path: (Path) Path to audio file.
    :param out_dir: (Path) Directory to write files.
    :param frame_len: (int) Length of frame to divide each audio file into.
    """

    audio_name = audio_path.stem
    source = init_audio(audio_path, hop_size=frame_len)
    audio_frames, sr = read_audio(source, read_frames=True)
    for index, frame in enumerate(audio_frames):
        np.save(out_dir / f"{audio_name}_{index}", frame)


def parallel_process_files(files: list[Path], out_dir: Path, frame_len: int):
    """
    Divides audio files into frames using multiprocessing.

    :param files: (Path) List of paths to audio files.
    :param out_dir: (Path) Directory to write files.
    :param frame_len: (int) Length of frame to divide each audio file into.
    """

    export_fn = partial(export_individual_frames, out_dir=out_dir, frame_len=frame_len)
    thread_map(export_fn, files, max_workers=os.cpu_count(), desc="Creating audio frames")


def process(in_dir: Path, out_dir: Path, frame_len: int):
    """
    Puts the path to each file in in_dir in a list, and calls parallel_process_files()
     to divide each file into frames using multiprocessing.

    :param files: (Path) Directory containing audio files to divide into frames.
    :param out_dir: (Path) Directory to write files.
    :param frame_len: (int) Length of frame to divide each audio file into.
    """

    files = list(in_dir.glob('*.wav'))

    start_time = time.perf_counter()
    parallel_process_files(files, out_dir, frame_len)
    finish_time = time.perf_counter()
    print(f'Finished dividing {len(files)} files in {finish_time-start_time} seconds.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', metavar='Audio directory', help="Directory containing audio to divide into frames")
    parser.add_argument('out_dir', metavar='Output directory', help="Directory to write audio frames")
    parser.add_argument('frame_len', metavar='Frame length', help="The length of each frame to divide audio into (should be the same as the intended frame_len for the annotations).", type=int)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    process(in_dir, out_dir, args.frame_len)