import random

from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from tools import audio_tools, meta_tools, misc_tools

class ADTOFDataset(Dataset):
    def __init__(self,
                 audio_files: list,
                 annotation_files: list,
                 frame_len: int):
        self.audio_files = audio_files
        self.annotation_files = annotation_files
        self.frame_len = frame_len


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
        audio_frames_np = audio_frames_np[:2756]
        audio_frames = torch.FloatTensor(audio_frames_np)

        # Read targets
        targets_frames_np = np.load(files[1])
        # TODO: Stop cropping like this and come up with a better way
        targets_frames_np = targets_frames_np[:2756]
        targets_frames = torch.from_numpy(targets_frames_np)

        assert files[0].stem == files[1].stem
        return audio_frames, targets_frames
    

def discard_missing_files(audio_files: list[Path], annotation_files: list[Path]):
    """
    Ensure that only audio files with respective annotations are loaded, and vice versa.

    :param audio_files: (list) List of audio file paths
    :param annotation_files: (list) List of annotation file paths

    :return: (Tuple) Input lists with missing files discarded
    """

    # TODO: Add a method for logging missing files
    audio_files = [file for file in audio_files if file.stem in [a_file.stem for a_file in annotation_files]]
    annotation_files = [file for file in annotation_files if file.stem in [a_file.stem for a_file in audio_files]]
    
    return audio_files, annotation_files
    

def test_train_valid_split(data_folder: Path, test_ratio: float, train_ratio: float, shuffle: bool = True):
    """
    Placeholder
    """

    # Prepare lists of files
    audio_path = data_folder / 'audio/audio/'
    audio_files = list(audio_path.glob('*.wav'))
    
    annotation_path = data_folder / 'annotations/one_hot/'
    annotation_files = list(annotation_path.glob('*.npy'))

    audio_files, annotation_files = discard_missing_files(audio_files, annotation_files)

    # Lists to NumPy
    audio_files = np.array(audio_files)
    annotation_files = np.array(annotation_files)

    split_audio_files = {}
    split_annotation_files = {}
    
    # Create test split
    # TODO: Make a script for creating a test split rather than doing this on the fly (further ensuring that this split is permanent)
    num_files = len(audio_files)
    num_test_files = round(num_files * test_ratio, 0)
    split_audio_files['test'], audio_files = np.split(audio_files, [int(len(audio_files) * test_ratio)])
    split_annotation_files['test'], annotation_files = np.split(annotation_files, [int(len(annotation_files) * test_ratio)])

    # Create train and valid split
    if shuffle:
        temp = list(zip(audio_files, annotation_files))
        random.shuffle(temp)
        audio_files, annotation_files = zip(*temp)
        # audio_files and annotation_files come out as tuples, and so must be converted to lists.
        audio_files, annotation_files = list(audio_files), list(annotation_files)
    split_audio_files['valid'], split_audio_files['train'] = np.split(audio_files, [int(len(audio_files) * test_ratio)])
    split_annotation_files['valid'], split_annotation_files['train'] = np.split(annotation_files, [int(len(annotation_files) * test_ratio)])

    return split_audio_files, split_annotation_files
    

def create_dataloaders(data_folder: Path, test_ratio: float, train_ratio: float, frame_len: int, batch_size: int, shuffle: bool = True):
    """
    Placeholder
    """
    
    # Split the dataset into test, train and valid sets of inputs and targets
    split_audio_files, split_annotation_files = test_train_valid_split(data_folder, test_ratio, train_ratio, shuffle)

    dataloaders = {}

    # Iterate over the split dataset
    for split in split_audio_files:
        audio_files = split_audio_files[split]
        annotation_files = split_annotation_files[split]

        dataset = ADTOFDataset(audio_files, annotation_files, frame_len)
        shuffle_split = shuffle if split != 'test' else False
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_split)

    return dataloaders