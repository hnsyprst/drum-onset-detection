from pathlib import Path

from torch.utils.data import Dataset

from tools import audio_tools, meta_tools, misc_tools

class ADTOFDataset(Dataset):
    def __init__(self,
                 data_folder: Path,
                 frame_len: int):
        self.data_folder = data_folder
        self.frame_len = frame_len

        # Set up audio file loading
        audio_path = self.data_folder / 'audio/audio/'
        self.audio_files = list(audio_path.glob('*.wav'))

        # Set up annotation loading
        annotation_path = self.data_folder / 'annotations/aligned_drum/'
        self.annotation_files = list(annotation_path.glob('*.txt'))

        # FIXME: A better way to handle this assert might be to check that every audio file has an annotation ahead of time
        assert len(self.audio_files) == len(self.annotation_files), f'Number of audio files and number of annotations must be the same. Got {len(self.audio_files)=}, {len(self.annotation_files)=}'


    def __len__(self):
        return len(self.audio_files)
    
    
    def __getitem__(self, idx):
        # TODO: Check that all matching audio files and annotations have the same indices, or work around this

        # Read audio
        source = audio_tools.in_out.init_audio(self.audio_files[idx], hop_size=self.frame_len)
        # FIXME: This should be a param of the class, and files outside the specified SR should be resampled or discarded
        audio_sr = source.samplerate
        audio_len = source.duration
        audio_frames = audio_tools.in_out.read_audio(source, read_frames=True)

        # Read annotations
        annotations = meta_tools.in_out.read_annotations_ADTOF(self.annotation_files[idx])
