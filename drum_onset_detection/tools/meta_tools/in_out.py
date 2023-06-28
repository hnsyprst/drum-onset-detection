import csv

import xml.etree.ElementTree as ET

from dataclasses import dataclass

from ..misc_tools.validation import validate_path


@dataclass(slots=True, frozen=True, kw_only=True)
class Annotation():
    """
    Data structure for storing annotations from IDMT-SMT-DRUMS-V2

    :pitch: (float | None) The drum's pitch (None if not applicable)
    :onset_sec: (float) The time at which the drum begins, in seconds.
    :instrument: (str) The name of the drum being played.
    """

    pitch: float | None
    onset_sec: float
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
                                      onset_sec=float(event[1].text),
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

    with open(path) as tsv:
        for line in csv.reader(tsv, delimiter="\t"):
            annotations.append(Annotation(pitch=None,
                                          onset_sec=float(line[0]),
                                          instrument=line[1]))
    return annotations