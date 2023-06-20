import os
import aubio

import numpy as np

from ..misc_tools.validation import validate_path

def init_audio(path: str, hop_size: int = 512) -> aubio.source:
    """
    Create an object that can read an audio file from the disc.
     This function uses aubio behind the scenes and returns an aubio.source object.
     This object can be called to read the next frame_size samples in the audio file from the disc,
     amongst other things (see aubio.source docs).
     If intending to read the entire file, you can pass this object to read_audio.
     It is recommended to close the returned object via source.close() when you are finished with it,
     but not required---the opened file will be automatically closed when the object falls out of scope.

    :param path: (str) The path to the audio file on the disc.
    :param hop_size: (int) The number of samples in the audio file to be read per iteration
    
    :return: (aubio.source) 
    """
    path = os.path.abspath(path)
    validate_path(path)
    source = aubio.source(path, hop_size=hop_size)
    return source


def read_audio(source: aubio.source, read_frames: bool = False, close: bool = True) -> tuple:
    """
    Read an audio file from the disc via an aubio.source object.

    Note that if `source.duration % source.hop_size != 0`,
     the remainder will be zero-padded.

    :param source: (aubio.source) An aubio.source object containing a reference to an audio file on the disc.
    :param read_frames: (bool) If True, return a matrix of `shape(N x source.hop_size)` frames.
                                Otherwise, return an array of `len(source.samples)`
    :param close: (bool) If true, call `source.close()` once reading is complete.

    :return: (tuple) 0. The audio file, as a Numpy array.
                     1. The audio file's sample rate.
    """
    N = (source.duration / source.hop_size).__floor__()
    shape = (N, source.hop_size)
    sr = source.samplerate
    audio = np.empty(shape, dtype=np.float32)

    for n in range(N):
        audio[n, :], _ = source()

    if not read_frames:
        audio = audio.ravel()

    if close:
        source.close()
    
    return audio, sr