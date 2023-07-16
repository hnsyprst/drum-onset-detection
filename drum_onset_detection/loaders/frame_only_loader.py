import random
import sys
import os

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
                 durations: pd.DataFrame,
                 load_mem: bool = False):
        self.audio_dir = audio_dir
        self.annotations_dir = annotations_dir
        self.frame_len = frame_len
        self.durations = durations
        self.load_mem = load_mem

        if self.load_mem:
            self.annotations_df = pd.DataFrame()
            print('Loading targets into memory...')
            # TODO: Figure out a better way of doing this; iteration is an antipattern in Pandas
            for file_name in durations['file_name']:
                target_np = np.load(annotations_dir / (file_name + '.npy')).astype(np.float32)
                self.annotations_df = pd.concat([self.annotations_df, pd.DataFrame(target_np)])
            print('Done!')

        
        # Calculate total number of frames in the dataset
        # TODO: 'cum_frames'.max()
        self.dataset_len = int(self.durations['duration'].sum() / frame_len)


    def __len__(self):
        return self.dataset_len
    
    
    def __getitem__(self, idx):
        # TODO: Find a way to get the index of the desired sample within the file
        #frame_start_sample_from_total = idx_to_frame(idx, self.frame_len)
        path_idx = self.durations['cum_frames'].searchsorted(idx, side='right')
        # if path_idx > 0:
        #     frame_start_sample = int((idx - self.durations['cum_frames'].iloc[path_idx - 1]) * self.frame_len)
        # else:
        #     frame_start_sample = int(idx * self.frame_len)

        previous_frames = int(self.durations['cum_frames'].iat[path_idx - 1] if path_idx > 0 else 0)
        frame_idx_in_file = int(idx - previous_frames)
        frame_start_sample = int(frame_idx_in_file * self.frame_len)

        # Read frame
        audio_path = self.audio_dir / (self.durations['file_name'].iloc[path_idx] + '.wav')
        source = audio_tools.in_out.init_audio(audio_path, hop_size=self.frame_len)
        audio_frame_np, sr = audio_tools.in_out.read_frame(source, frame_start_sample)
        audio_frame = torch.FloatTensor(audio_frame_np)

        # Read target
        if self.load_mem:
            target_frame_np = self.annotations_df.iloc[idx]
            target_frame = torch.tensor(target_frame_np.values, dtype=torch.float32)
        else:
            target_path = self.annotations_dir / (self.durations['file_name'].iloc[path_idx] + '.npy')
            target_np = np.load(target_path).astype(np.float32)
            frame_idx = int(frame_start_sample / self.frame_len)
            target_frame_np = target_np[frame_idx_in_file]
            target_frame = torch.from_numpy(target_frame_np)

        return audio_frame, target_frame

    
def idx_to_frame(idx, frame_len):
    """
    Placeholder
    """
    return int((idx + 1) * frame_len)
    

def prepare_durations_dataframe(durations: pd.DataFrame, frame_len: int):
    """
    Prepares durations dataframe for dataloader use.

    :param metadata_dir: (Path) Directory containing 'durations.csv'.
    :param frame_len: (int) Length of frame to divide each audio file into.

    :return: (pd.DataFrame) Dataframe formatted for dataloader use. 
    """

    # Round all durations up to the nearest frame_len, as frames will be zero-padded when read
    durations['duration'] = durations['duration'].apply(lambda x: np.ceil(x / frame_len) * frame_len)
    durations['num_frames'] = durations['duration'].apply(lambda x: x / frame_len)
    # Create cumulative sum column
    durations['cum_duration'] = durations['duration'].cumsum()
    durations['cum_frames'] = durations['num_frames'].cumsum()

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

    # Load durations metadata
    durations = pd.read_csv(metadata_dir / 'durations.csv', names=['file_name', 'duration'])
    durations = discard_missing_files(durations, audio_dir, annotations_dir)
    durations = prepare_durations_dataframe(durations, frame_len)
    
    # Split the dataset into test, train and valid sets of inputs and targets
    split_durations = test_train_valid_split(durations, frame_len, test_ratio, train_ratio, shuffle)

    dataloaders = {}

    # Iterate over the split dataset
    for split, durations in split_durations.items():
        dataset = ADTOFFramesDataset(audio_dir, annotations_dir, frame_len, durations, load_mem=True)
        shuffle_split = shuffle if split != 'test' else False
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_split, num_workers=os.cpu_count())

    return dataloaders