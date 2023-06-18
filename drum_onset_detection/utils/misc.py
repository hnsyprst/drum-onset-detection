import numpy as np

# TODO: Implement - could either dither or just add a tiny DC signal
def handle_silence(audio: np.array) -> np.array:
    """
    Prevents NaNs by ensuring no zeroes remain in an audio file.

    :param audio: (np.array) An array of audio samples.
    """
    pass