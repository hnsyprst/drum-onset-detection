import argparse
import multiprocessing
import os
import sys

import pandas as pd

from pathlib import Path
from tqdm import tqdm

sys.path.append('..')
from drum_onset_detection.tools import audio_tools, meta_tools

def get_bpm(out_dict: dict, path: Path | str):
    """
    Initialises an audio source from a path and calculates the audio's BPM.

    :param out_dict: (dict) Dictionary to append calculated BPM
    :param path: (Path | str) Path to an audio file.
    """

    source = audio_tools.in_out.init_audio(path)
    bpm = meta_tools.analysis.get_bpm(source)
    out_dict[path] = bpm

def _update(*path):
    """
    Callback function for updating tqdm progress bar.
    """

    pbar.update()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', metavar='Data folder', help="Directory containing files to analyse")
    parser.add_argument('output_path', metavar='Output path', help="Path to write BPMs")
    args = parser.parse_args()

    data_folder = Path(args.data_folder)
    # Create a progress bar by getting the total number of audio files to analyse
    pbar = tqdm(total=len(list(data_folder.glob('*'))))

    # Calculate BPMs using multiprocessing
    manager = multiprocessing.Manager()
    bpms = manager.dict()
    pool = multiprocessing.Pool(os.cpu_count())
    for path in data_folder.iterdir():
        pool.apply_async(get_bpm, args=(bpms, path,), callback=_update)
    pool.close()
    pool.join()
    print("Finished BPM calculation")

    # Write BPMs to file
    df = pd.DataFrame.from_dict(bpms, orient='index')
    df.to_csv(args.output_path, header=False)