import random
import sys

from pathlib import Path

import numpy as np
import pandas as pd

import torch
import librosa
from torch.utils.data import Dataset, DataLoader

sys.path.append('..')
from tools import audio_tools, meta_tools, misc_tools
    

class ADTOFFramesDataset(Dataset):
    def __init__(self,
                 audio_dir: Path,
                 annotations_dir: Path,
                 frame_len: int,
                 durations: pd.DataFrame):
        self.audio_dir = audio_dir
        self.annotations_dir = annotations_dir
        self.frame_len = frame_len
        self.durations = durations
        
        # Calculate total number of frames in the dataset
        self.dataset_len = int(self.durations['duration'].sum() / frame_len)


    def __len__(self):
        return self.dataset_len
    
    
    def __getitem__(self, idx):
        # TODO: Find a way to get the index of the desired sample within the file
        frame_start_sample_from_total = idx_to_frame(idx, self.frame_len)
        path_idx = self.durations['cum_duration'].searchsorted(frame_start_sample_from_total, 'right')
        frame_start_sample_in_file = int(self.durations['cum_duration'].iloc[path_idx] - frame_start_sample_from_total)

        # Read frame
        audio_path = self.audio_dir / (self.durations['file_name'].iloc[path_idx + 1] + '.wav')
        source = audio_tools.in_out.init_audio(audio_path, hop_size=self.frame_len)
        audio_frame_np, sr = audio_tools.in_out.read_frame(source, frame_start_sample_in_file)
        audio_frame = torch.FloatTensor(audio_frame_np)

        # Read target
        target_path = self.annotations_dir / (self.durations['file_name'].iloc[path_idx + 1] + '.npy')
        target_np = np.load(target_path).astype(np.float32)
        frame_idx = int(frame_start_sample_in_file / self.frame_len)
        target_frame_np = target_np[frame_idx]
        target_frame = torch.from_numpy(target_frame_np)

        return audio_frame, target_frame
    
def idx_to_frame(idx, frame_len):
    """
    Placeholder
    """
    return int((idx + 1) * frame_len)
    

def prepare_durations_dataframe(metadata_dir: Path, frame_len: int):
    """
    Load in a CSV file called 'durations.csv' in the directory metadata_dir and prepares it
     for dataloader use.

    :param metadata_dir: (Path) Directory containing 'durations.csv'.
    :param frame_len: (int) Length of frame to divide each audio file into.

    :return: (pd.DataFrame) Dataframe formatted for dataloader use. 
    """

    # Load durations metadata
    durations = pd.read_csv(metadata_dir / 'durations.csv', names=['file_name', 'duration'])
    # Round all durations up to the nearest frame_len, as frames will be zero-padded when read
    durations['duration'] = durations['duration'].apply(lambda x: np.ceil(x / frame_len) * frame_len)
    # Create cumulative sum column
    durations['cum_duration'] = durations['duration'].cumsum()

    return durations
    

def discard_missing_files(durations: pd.DataFrame, audio_dir: Path, annotations_dir: Path):
    """
    Ensure that only audio files with respective annotations are loaded, and vice versa.

    :param durations: (pd.DataFrame) Placeholder
    :param audio_dir: (Path) Placeholder
    :param annotations_dir: (Path) Placeholder

    :return: Placeholder
    """
    # TODO: Add a method for logging missing files

    def paths_exist(file_name):
        return (audio_dir / (file_name + '.wav')).exists() and (annotations_dir / (file_name + '.npy')).exists()

    return durations[durations['file_name'].apply(paths_exist)]
    

def test_train_valid_split(durations: pd.DataFrame, frame_len: int, test_ratio: float, train_ratio: float, shuffle: bool = True):
    """
    Placeholder
    """
    
    dataset_len = durations['duration'].sum() / frame_len

    # Create test split
    num_test_frames = dataset_len * test_ratio
    final_test_frame_start_sample = idx_to_frame(num_test_frames, frame_len)
    test_cutoff_idx = durations['cum_duration'].searchsorted(final_test_frame_start_sample)

    test_durations = durations.iloc[:test_cutoff_idx]
    train_valid_durations = durations.iloc[test_cutoff_idx:]

    # Create train and valid splits
    num_train_frames = dataset_len * train_ratio
    final_train_frame_start_sample = idx_to_frame(num_train_frames, frame_len)
    train_cutoff_idx = durations['cum_duration'].searchsorted(final_train_frame_start_sample)

    train_durations = durations.iloc[:train_cutoff_idx]
    valid_durations = durations.iloc[train_cutoff_idx:]

    return {'test': test_durations, 'train': train_durations, 'valid': valid_durations}

    

def create_dataloaders(data_folder: Path, test_ratio: float, train_ratio: float, frame_len: int, batch_size: int, shuffle: bool = True):
    """
    Placeholder
    """

    # Prepare paths
    audio_dir = data_folder / 'audio/audio/'
    annotations_dir = data_folder / 'annotations/one_hot/' / str(frame_len)
    metadata_dir = data_folder / 'audio/'

    durations = prepare_durations_dataframe(metadata_dir, frame_len)
    durations = discard_missing_files(durations, audio_dir, annotations_dir)
    
    # Split the dataset into test, train and valid sets of inputs and targets
    split_durations = test_train_valid_split(durations, frame_len, test_ratio, train_ratio, shuffle)

    dataloaders = {}

    # Iterate over the split dataset
    for split, durations in split_durations.items():
        dataset = ADTOFFramesDataset(audio_dir, annotations_dir, frame_len, durations)
        shuffle_split = shuffle if split != 'test' else False
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_split)

    return dataloaders