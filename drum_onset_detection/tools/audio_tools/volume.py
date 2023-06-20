import numpy as np

def fade_out(audio: np.array, duration: int) -> np.array:
    """
    Fades audio to 0 over the final `duration` samples.
     Audio must be 1 dimensional.

    :param audio: (np.array) 1D array of audio samples.
    :param duration: (int) Fade out duration, in samples.

    :return: Faded audio.
    """
    fade_curve = np.linspace(1.0, 0.0, duration)
    audio[:duration] = audio[:duration] * fade_curve

    return audio