def seconds_to_samples(seconds: float, sr: int):
    """
    Convert a time in seconds, assumed to be relative to 0 seconds,
     to a number of samples elapsed since 0 seconds, given a specified
     sampling rate.
    
    :param seconds: (float) A time in seconds.
    :param sr: The sampling rate, in Hz.

    :return: The number of samples elapsed at the given time.
    """

    return round(seconds * sr)