import random

from pathlib import Path

import numpy as np
import pandas as pd

import torch
import librosa
from torch.utils.data import Dataset, DataLoader

from tools import audio_tools, meta_tools, misc_tools
    

class ADTOFFramesDataset(Dataset):
    def __init__(self,
                 audio_dir: Path,
                 annotation_dir: Path,
                 frame_len: int,
                 durations: pd.DataFrame):
        self.audio_dir = audio_dir
        self.annotation_dir = annotation_dir
        self.frame_len = frame_len
        self.durations = durations
        
        # Calculate total number of frames in the dataset
        self.dataset_len = self.durations['duration'].sum() / frame_len


    def __len__(self):
        return self.dataset_len
    

    def idx_to_frame(self, idx):
        return (idx + 1) * self.frame_len
    
    
    def __getitem__(self, idx):
        frame_start_sample = self.idx_to_frame(idx)
        audio_path_idx = self.durations['cum_duration'].searchsorted(frame_start_sample)
        audio_path = self.audio_dir / self.durations['file_name'].iloc[audio_path_idx]
        # Read frame
        source = audio_tools.in_out.init_audio(audio_path, hop_size=self.frame_len)
        audio_frame_np, sr = audio_tools.in_out.read_frame(source, frame_start_sample)
        audio_frame = torch.FloatTensor(audio_frame_np)

        # Read target
        target_path = self.targets_dir / self.durations['file_name'].iloc[audio_path_idx]
        target_frame_np = np.load(target_path).astype(np.float32)
        target_frame = torch.from_numpy(target_frame_np)

        return audio_frame, target_frame
    

def prepare_durations_dataframe(metadata_dir: Path, frame_len: int):
    """
    
    """
    # Load durations metadata
    durations = pd.read_csv(metadata_dir / 'durations.csv', names=['file_name', 'duration'])
    # Round all durations up to the nearest frame_len, as frames will be zero-padded when read
    durations['duration'] = durations['duration'].apply(lambda x: np.ceil(x / frame_len) * frame_len)
    # Create cumulative sum column
    durations['cum_duration'] = durations['duration'].cumsum()

    return durations
    

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

        dataset = ADTOFFramesDataset(audio_files, annotation_files, frame_len)
        shuffle_split = shuffle if split != 'test' else False
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_split)

    return dataloaders