from aubio import source, tempo
from numpy import median, diff

def get_bpm(source: source):
    """
    Calculate the beats per minute (bpm) of a given audio track.

    :param source: (aubio.source) aubio.source object (try audio_tools.in_out.init_audio())
    """
    t = tempo("specdiff", samplerate=source.samplerate)
    # List of beats, in samples
    beats = []
    # Total number of frames read
    total_frames = 0

    while True:
        samples, read = source()
        is_beat = t(samples)
        if is_beat:
            this_beat = t.get_last_s()
            beats.append(this_beat)
            #if o.get_confidence() > .2 and len(beats) > 2.:
            #    break
        total_frames += read
        if read < source.hop_size:
            break

    def beats_to_bpm(beats, source):
        # if enough beats are found, convert to periods then to bpm
        if len(beats) > 1:
            if len(beats) < 4:
                print("few beats found in {:s}".format(source.uri))
            bpms = 60./diff(beats)
            return median(bpms)
        else:
            print("not enough beats found in {:s}".format(source.uri))
            return 0

    return beats_to_bpm(beats, source)