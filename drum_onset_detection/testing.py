import sys

sys.path.append('..')
import utils

source = utils.io.init_audio(r'F:\Work2\drum-onset-detection\data\IDMT-SMT-DRUMS-V2\audio\RealDrum01_00#HH#train.wav')
audio, sr = utils.io.read_audio(source)
