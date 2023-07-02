from pathlib import Path

import numpy as np

import torch
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
        annotation_path = self.data_folder / 'annotations/one_hot/'
        self.annotation_files = list(annotation_path.glob('*.npy'))

        # Ensure that only audio files with respective annotations are loaded
        self.audio_files = [file for file in self.audio_files if file.stem in [a_file.stem for a_file in self.annotation_files]]
        # Ensure that only annotations with respective audio files are loaded
        self.annotation_files = [file for file in self.annotation_files if file.stem in [a_file.stem for a_file in self.audio_files]]
        # FIXME: Add a function to log missing files
        
        # FIXME: A better way to handle this assert might be to check that every audio file has an annotation ahead of time
        assert len(self.audio_files) == len(self.annotation_files), f'Number of audio files and number of annotations must be the same. Got {len(self.audio_files)=}, {len(self.annotation_files)=}'


    def __len__(self):
        return len(self.audio_files)
    
    
    def __getitem__(self, idx):
        # TODO: Check that all matching audio files and annotations have the same indices, or work around this

        files = (self.audio_files[idx], self.annotation_files[idx])

        # Read audio
        source = audio_tools.in_out.init_audio(files[0], hop_size=self.frame_len)
        # FIXME: This should be a param of the class, and files outside the specified SR should be resampled or discarded
        audio_sr = source.samplerate
        audio_frames_np, sr = audio_tools.in_out.read_audio(source, read_frames=True)
        audio_frames = torch.FloatTensor(audio_frames_np)

        # Read targets
        targets_frames_np = np.load(files[1])
        targets_frames = torch.from_numpy(targets_frames_np)

        assert files[0].stem == files[1].stem

        return audio_frames, targets_frames