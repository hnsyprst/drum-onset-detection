from dataclasses import dataclass
import xml.etree.ElementTree as ET

from ..misc_tools.validation import validate_path

@dataclass(slots=True, frozen=True, kw_only=True)
class Annotation():
    """
    Data structure for storing annotations from IDMT-SMT-DRUMS-V2

    :pitch: (float) The drum's pitch.
    :onset_sec: (float) The time at which the drum begins, in seconds.
    :offset_sec: (float) The time at which the drum ends, in seconds.
    :instrument: (str) The name of the drum being played.
    """

    pitch: float
    onset_sec: float
    offset_sec: float
    instrument: str


def read_annotations(path: str):
    """
    Read IDMT-SMT-DRUMS-V2 annotations from an XML file.

    :param path: (str) The path to the annotations file on the disc.
    
    :return: (list[Annotation]) A list of Annotations read from the file.
    """

    validate_path(path)
    tree = ET.parse(path)
    root = tree.getroot()

    annotations = []

    for event in root.iter('event'):
        annotations.append(Annotation(pitch=float(event[0].text),
                                      onset_sec=float(event[1].text),
                                      offset_sec=float(event[2].text),
                                      instrument=event[3].text))
    return annotations