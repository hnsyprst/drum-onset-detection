import argparse
import time
import os
import sys

import numpy as np
import pandas as pd

from tqdm.contrib.concurrent import thread_map
from pathlib import Path
from functools import partial

sys.path.append('..')
from drum_onset_detection.tools.audio_tools.in_out import init_audio

def calculate_duration(audio_path: Path, durations: dict):
    """
    Load an audio file at audio_path, calculate the file's duration, and append
     this information to the durations dictionary.

    :param audio_path: (Path) Path to audio file.
    :param durations: (Path) Dictionary to add calculated duration to.
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

    durations = {}
    duration_fn = partial(calculate_duration, durations=durations)
    thread_map(duration_fn, files, max_workers=os.cpu_count(), desc="Calculating durations")
    return durations


def process(in_dir: Path, out_path: Path):
    """
    Puts the path to each file in in_dir in a list, and calls parallel_process_files()
     to calculate the duration of each file into using multiprocessing.

    :param in_dir: (Path) Directory containing audio to process.
    :param out_path: (Path) Path to write metadata.
    """

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

    in_dir = Path(args.in_dir)
    out_path = Path(args.out_path)

    process(in_dir, out_path)