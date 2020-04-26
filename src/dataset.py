import os
import json
import h5py
import torch
import random
import numpy as np
from torch.utils.data import Dataset

class EvaluateDateset(Dataset):
    def __init__(self, h5_path, speakers, utterances, segment_size, load_spectrogram='mel'):
        self.dataset = None
        self.h5_path = h5_path
        self.speakers = speakers
        self.utterances = utterances
        self.segment_size = segment_size
        self.load_spectrogram = load_spectrogram

    def __getitem__(self,index):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, 'r')

        data_lin = self.dataset[f'{self.speakers[index]}/{self.utterances[index]}/lin'][:]
        data_mel = self.dataset[f'{self.speakers[index]}/{self.utterances[index]}/mel'][:].transpose(1,0)
        data_dmel = self.dataset[f'{self.speakers[index]}/{self.utterances[index]}/dmel'][:]

        if self.segment_size is not None:
            t = random.randint(0, len(data_lin) - self.segment_size)
            data_lin = data_lin[t:t+self.segment_size]
            data_mel = data_mel[t:t+self.segment_size]
        

        if self.load_spectrogram == 'mel':
            return {'speaker': self.speakers[index], 'spectrogram': data_mel}
        elif self.load_spectrogram == 'dmel':
            return {'speaker': self.speakers[index], 'spectrogram': data_dmel}

    def __len__(self):
        return len(self.speakers)

    def collate_fn(self, datas):
        batch = {}
        batch['speaker'] = [data['speaker'] for data in datas]
        batch['len'] = [len(data['spectrogram']) for data in datas]
        batch['spectrogram'] = torch.tensor([self.pad_to_len(data['spectrogram'], max(batch['len'])) for data in datas])
        return batch

    def pad_to_len(self, arr, padded_len):
        padded_len = int(padded_len)
        padding = np.expand_dims(np.zeros_like(arr[0]),0)
        if (len(arr) < padded_len):
            for i in range(padded_len-len(arr)):
                arr = np.concatenate((arr,padding),axis=0)
        return arr[:padded_len]


class TransferDateset(Dataset):
    def __init__(self, h5_path, speakers, utterances, indexes, segment_size):
        self.dataset = None
        self.h5_path = h5_path
        self.speakers = speakers
        self.utterances = utterances
        self.segment_size = segment_size
        self.indexes = indexes

    def __getitem__(self,index):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, 'r')

        data_lin = self.dataset[f'{self.speakers[index]}/{self.utterances[index]}/lin'][:]
        data_mel = self.dataset[f'{self.speakers[index]}/{self.utterances[index]}/mel'][:].transpose(1,0)
        data_dmel = self.dataset[f'{self.speakers[index]}/{self.utterances[index]}/dmel'][:]

        random_speaker =  random.sample(set(list(self.indexes.keys())) - set(self.speakers[index]), 1)[0]
        random_utt = random.sample(self.indexes[random_speaker],1)[0]
        random_mel = self.dataset[f'{random_speaker}/{random_utt}/mel'][:].transpose(1,0)

        return {'speaker': self.speakers[index], 'tar': data_mel, 'tar_dmel':data_dmel, 'src': random_mel}


    def collate_fn(self, datas):
        batch = {}
        batch['speaker'] = [data['speaker'] for data in datas]
        batch['len'] = [len(data['spectrogram']) for data in datas]
        batch['spectrogram'] = torch.tensor([self.pad_to_len(data['spectrogram'], max(batch['len'])) for data in datas])
        batch['len'] = [len(data['source']) for data in datas]
        batch['source'] = torch.tensor([self.pad_to_len(data['source'], max(batch['len'])) for data in datas])
        return batch

    def pad_to_len(self, arr, padded_len):
        padded_len = int(padded_len)
        padding = np.expand_dims(np.zeros_like(arr[0]),0)
        if (len(arr) < padded_len):
            for i in range(padded_len-len(arr)):
                arr = np.concatenate((arr,padding),axis=0)
        return arr[:padded_len]

    def __len__(self):
        return len(self.speakers)


class TotalDataset(Dataset):
    def __init__(self, h5_path, index_path, speaker_info_path, emotion, segment_size=128):
        self.dataset = None
        self.h5_path = h5_path
        self.emotion = emotion

        # Split with different speaker
        with open(index_path) as f:
            self.class_indexes = json.load(f)

        self.speaker_emotions = {}
        with open(speaker_info_path, 'r') as f:
            for i, line in enumerate(f):
                speaker_id = line.strip().split()[0]
                speaker_emotion = line.strip().split()[1]
                self.speaker_emotions[speaker_id] = speaker_emotion

        # All data
        self.total_indexes = []
        for speaker_id in self.class_indexes:
            if self.speaker_emotions[speaker_id] == emotion:
                self.total_indexes += [speaker_id]

        print('Total Utterances {}: {}'.format(emotion, len(self.total_indexes)))

        self.segment_size = segment_size

    def __getitem__(self, index):

        # if self.dataset is None:
            # self.dataset = h5py.File(self.h5_path, 'r')

        speaker_id = self.total_indexes[index]

        with h5py.File(self.h5_path, 'r') as file:
            mel = file['{}/mel'.format(speaker_id)][:]
        
        data = {
            'speaker': speaker_id,  
            'mel': mel
            # N * 160 * 40
        }
        
        # print(data['mel'].shape)


        if self.segment_size is not None:
            t = random.randint(0, len(data['mel']) - self.segment_size)
            data['mel'] = data['mel'][t:t+self.segment_size]
                
        return data

    def __len__(self):
        return len(self.total_indexes)

