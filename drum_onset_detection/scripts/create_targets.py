import argparse
import multiprocessing
import os
import sys

import numpy as np

from pathlib import Path
from functools import partial
from tqdm import tqdm

sys.path.append('..')
from drum_onset_detection.tools import audio_tools, meta_tools

def get_one_hot(classes: list, frame_len: int, annotations_path: Path | str, audio_path: Path | str, out_path: Path | str):
    """
    Reads ADTOF annotations from a path and write a one-hot representation to the disc.

    :param classes: (list) List of instrument names from the dataset.
    :param frame_len: (int) The length of each frame to divide annotations into (should be the same as the intended frame_len for the audio).
    :param annotations_path: (Path | str) Path to an annotations file.
    :param audio_path: (Path | str) Path to the corresponding audio file.
    :param out_path: (Path | str) Path to write the constructed one-hot representation.
    """
    # TODO: Handle NaNs / Nones
    # Get audio duration
    source = audio_tools.in_out.init_audio(audio_path)
    duration = source.duration
    sr = source.samplerate

    # Convert annotations
    annotations = meta_tools.in_out.read_annotations_ADTOF(annotations_path)
    annotation_frames = meta_tools.in_out.annotations_list_to_frames(annotations, duration, frame_len, sr)
    one_hot = meta_tools.in_out.annotation_frames_to_one_hot(annotation_frames, classes)
    np.save(out_path, one_hot)


def _update(*path):
    """
    Callback function for updating tqdm progress bar.
    """

    pbar.update()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annotations_dir', metavar='Annotations directory', help="Directory containing annotations to convert")
    parser.add_argument('audio_dir', metavar='Audio directory', help="Directory containing corresponding audio files for annotations")
    parser.add_argument('output_dir', metavar='Output directory', help="Directory to write dataset")
    parser.add_argument('classes', metavar='Classes', help="List of instrument classes present in the dataset")
    parser.add_argument('frame_len', metavar='Frame length', help="The length of each frame to divide annotations into (should be the same as the intended frame_len for the audio).", type=int)
    args = parser.parse_args()

    # Confirm inputs
    classes = args.classes.split(",")
    print('Confirming inputs---is the following data correct?')
    print(f'Classes: {classes}')
    print(f'Frame length: {args.frame_len}')
    input('Press enter to continue (or keyboard interrupt to exit)')

    annotations_dir = Path(args.annotations_dir)
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    # Create a progress bar by getting the total number of annotations
    pbar = tqdm(total=len(list(annotations_dir.glob('*'))))

    # Generate one-hot representations using multiprocessing
    # manager = multiprocessing.Manager()
    # one_hot = manager.dict()
    pool = multiprocessing.Pool(os.cpu_count())
    for annotations_path in annotations_dir.iterdir():
        file_name = annotations_path.stem
        audio_path = audio_dir / f'{file_name}.wav'
        out_path = output_dir / f'{file_name}.npy'
        pool.apply_async(get_one_hot, args=(classes, args.frame_len, annotations_path, audio_path, out_path), callback=_update)
    pool.close()
    pool.join()
    print("Finished creating one-hot targets")