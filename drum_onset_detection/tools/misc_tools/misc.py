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


def get_frame_index(sample: int, duration: int, frame_len: int):
    """
    Given a sample within a duration to be divided into frames of length frame_len, find the index
    of the frame containing the given sample.

    :param sample: (int) Sample to locate.
    :param duration: (int) Duration containing sample.
    :param frame_len: (int) The length of each frame that duration will be divided into.

    :return: (int) The index of the frame containing the given sample.
    """

    assert sample <= duration, f"Sample must be within duration. Got {sample} > {duration}."
    # Get number of frames to divide into by ceil div
    n_frames = -(duration // -frame_len)
    return 1 + sample // (duration // n_frames) if sample >= 0 else 0