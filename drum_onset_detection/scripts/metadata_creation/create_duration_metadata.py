import argparse
import time
import os
import sys
import threading

import numpy as np
import pandas as pd

from tqdm.contrib.concurrent import thread_map
from pathlib import Path
from functools import partial

sys.path.append('..')
from drum_onset_detection.tools.audio_tools.in_out import init_audio, read_audio

def calculate_duration(audio_path: Path, durations: dict):
    """
    Load an audio file at audio_path, divide into into frames of length frame_len,
     and save each frame as an individual .npy file.

    :param audio_path: (Path) Path to audio file.
    :param out_dir: (Path) Directory to write files.
    """

    audio_name = audio_path.stem
    source = init_audio(audio_path, hop_size=512)
    # TODO: Could add calculation here for number of frames in each file; this might speed up initialisation of the dataset
    durations[audio_name] = source.duration


def parallel_process_files(files: list[Path]):
    """
    Calculates the durations of each audio file in a list using multiprocessing.

    :param files: (Path) List of paths to audio files.
    """

    # TODO: Add results of calculate_duration to a list and return the final list from this function
    durations = {}
    duration_fn = partial(calculate_duration, durations=durations)
    thread_map(duration_fn, files, max_workers=os.cpu_count(), desc="Calculating durations")
    return durations


def process(in_dir: Path, out_path: Path):
    """
    Puts the path to each file in in_dir in a list, and calls parallel_process_files()
     to calculate the duration of each file into using multiprocessing.

    :param files: (Path) List of paths to audio files.
    """

    # TODO: Write results from parallel_process_files to a file at out_path
    files = list(in_dir.glob('*.wav'))

    start_time = time.perf_counter()
    durations = parallel_process_files(files)
    finish_time = time.perf_counter()
    print(f'Finished calculating the duration of {len(files)} files in {finish_time-start_time} seconds.')
    print(f'Writing results to file...')
    # Write durations to file
    df = pd.DataFrame.from_dict(durations, orient='index')
    df.to_csv(out_path, header=False)
    print(f'Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', metavar='Audio directory', help="Directory containing audio to process")
    parser.add_argument('out_path', metavar='Output path', help="Path to write metadata")
    args = parser.parse_args()

    # global lock
    # lock = threading.lock()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out_path)

    process(in_dir, out_path)