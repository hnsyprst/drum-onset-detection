import csv

import xml.etree.ElementTree as ET

from dataclasses import dataclass

import numpy as np

from ..misc_tools.validation import validate_path
from ..misc_tools.misc import seconds_to_samples, get_frame_index


@dataclass(slots=True, frozen=True, kw_only=True)
class Annotation():
    """
    Data structure for storing annotations from IDMT-SMT-DRUMS-V2

    :pitch: (float | None) The drum's pitch (None if not applicable)
    :onset: (float) The time at which the drum begins, in seconds.
    :instrument: (str) The name of the drum being played.
    """

    pitch: float | None
    onset: float
    instrument: str


def read_annotations_IDMT(path: str):
    """
    Read IDMT-SMT-DRUMS-V2 annotations from an XML file.

    :param path: (str) The path to the annotations file on the disc.
    
    :return: (list[Annotation]) A list of Annotations read from the file.
    """

    validate_path(path)
    annotations = []

    tree = ET.parse(path)
    root = tree.getroot()

    for event in root.iter('event'):
        annotations.append(Annotation(pitch=float(event[0].text),
                                      onset=float(event[1].text),
                                      instrument=event[3].text))
    return annotations


def read_annotations_ADTOF(path: str):
    """
    Read ADTOF annotations from a tab-separated TXT file.

    :param path: (str) The path to the annotations file on the disc.
    
    :return: (list[Annotation]) A list of Annotations read from the file.
    """

    validate_path(path)
    annotations = []

    midi_map = {'35': 'KD',
                '38': 'SD',
                '42': 'HH',
                '47': 'TT',
                '49': 'CC'}

    with open(path) as tsv:
        for line in csv.reader(tsv, delimiter="\t"):
            annotations.append(Annotation(pitch=None,
                                          onset=float(line[0]),
                                          instrument=midi_map[line[1]]))
    return annotations


def construct_annotation_matrix(annotations: list, audio: np.array, sr: int):
    """
    Construct a binary matrix to represent the presence of annotations
     at each sample in a file.

    :param annotations: (list) Placeholder
    :param audio: (np.array) Placeholder
    :param sr: (int) The sampling rate, in Hz.

    :return: Placeholder
    """
    
    # Get each unique instrument in our annotation
    instruments = np.unique([annotation.instrument for annotation in annotations])
    num_instruments = len(instruments)
    index_instrument_mapping = dict(enumerate(instruments))

    # Construct an empty array, the same size as our input audio
    annotation_matrix = np.zeros_like(audio, dtype=bool)
    # Stack this array `num_instruments` times, over the first axis
    # We now have a matrix, in which each instrument has a single row of annotations
    annotation_matrix = np.stack([annotation_matrix] * num_instruments, axis=0)
    # We will populate each cell (or sample) in each row with `True` values
    # if an Annotation is reporting an onset of that row's instrument at that sample
    # and otherwise leave these values `False`.

    # For each unique instrument...
    for instrument_index, instrument in enumerate(instruments):
        # ... get all of the Annotations containing this instrument
        instrument_annotations = (annotation for annotation in annotations if annotation.instrument == instrument)
        # For each Annotation containing our instrument...
        for annotation in instrument_annotations:
            # ... get the Annotation's onset in samples and set this sample `True`
            onset_samples = seconds_to_samples(annotation.onset, sr)
            annotation_matrix[instrument_index][onset_samples] = True
    return annotation_matrix, index_instrument_mapping


def annotations_list_to_frames(annotations: list[Annotation], duration: int, frame_len: int, sr: int):
    """
    Divide a list of annotations into a list of frames containing annotations.
     This is so that audio frames and annotation frames match up.

    :param annotations: (list[Annotation]) A list of annotations.
    :param duration: (int) Duration of annotated audio.
    :param frame_len: (int) The length of each frame that duration will be divided into.
    :param sr: (int) Sampling rate of the annotated audio.

    :return: (list[list[Annotation | None]]) The annotations divided into frames.
    """
    
    # Get number of frames to divide into by ceil div
    n_frames = -(duration // -frame_len)
    annotation_frames = [[] for _ in range(n_frames)]

    for annotation in annotations:
        onset_samples = seconds_to_samples(annotation.onset, sr)
        frame_index = get_frame_index(onset_samples, duration, frame_len)
        annotation_frames[frame_index].append(annotation)

    return annotation_frames


def annotation_frames_to_one_hot(annotation_frames: list[list[Annotation | None]], classes: list):
    """
    Construct a one-hot representation of a list of frames containing annotations.

    :param annotation_frames: (list[list[Annotation | None]]) A list of frames containing annotations.
    :param classes: (list) List of possible classes, in the order they will appear in the one-hot representation.
    """

    one_hot = []
    n_classes = len(classes)
    # Convert list of classes to dict for O(1) mapping of class name to one-hot index
    classes = {class_name: idx for idx, class_name in enumerate(classes)}

    # For each annotation frame...
    for annotation_frame in annotation_frames:
        # ... Create an empty one-hot representation
        frame = [0 for _ in range(n_classes)]
        # If there are annotations in this frame, amend the empty one-hot representation
        if annotation_frame is not None:
            for annotation in annotation_frame:
                frame[classes[annotation.instrument]] = 1
        # append the final one-hot representation to the list
        one_hot.append(frame)

    return one_hot