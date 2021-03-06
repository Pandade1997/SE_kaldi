# dataset_path = "/data01/AuFast/PanJiaHui/LibriSpeech/SE_kaldi_LibriSpeech"
# dataset_path = "/data01/AuFast/Pan_dataset/SE_asr/test1/se_dataset"
# dataset_path = "/data01/AuFast/Pan_dataset/SE_asr/finaltest/wsj0"
# dataset_path = "/data01/AuFast/Pan_dataset/SE_asr/finaltest/libri"


# -*- coding: utf-8 -*-
import os
import numpy as np
from tqdm import tqdm
import librosa
import torch
from torch.utils import data

# dataset_path = "/data01/AuFast/PanJiaHui/LibriSpeech/SE_kaldi_LibriSpeech"
dataset_path = "/data01/AuFast/PanJiaHui/wsj_timit"


# DATA LOADING - LOAD FILE LISTS
def load_data_list(folder=dataset_path, setname='train'):
    assert (setname in ['train', 'valid', 'test'])

    dataset = {}

    if "test" in setname:
        clean_foldername = folder + '/testset'
    else:
        clean_foldername = folder + '/' + setname + "set"
    noisy_foldername = folder + '/' + setname + "set"

    print("Loading files...")
    dataset['innames'] = []
    dataset['outnames'] = []
    dataset['shortnames'] = []

    noisy_filelist = os.listdir("%s_noisy" % (noisy_foldername))
    noisy_filelist.sort()
    # filelist = [f for f in filelist if f.endswith(".wav")]
    for i in tqdm(noisy_filelist):
        dataset['innames'].append("%s_noisy/%s" % (noisy_foldername, i))
        dataset['shortnames'].append("%s" % (i))

    clean_filelist = os.listdir("%s_clean" % (clean_foldername))
    clean_filelist.sort()
    for i in tqdm(clean_filelist):
        dataset['outnames'].append("%s_clean/%s" % (clean_foldername, i))

    return dataset


# DATA LOADING - LOAD FILE DATA
def load_data(dataset):
    dataset['inaudio'] = [None] * len(dataset['innames'])
    dataset['outaudio'] = [None] * len(dataset['outnames'])

    for id in tqdm(range(len(dataset['innames']))):

        if dataset['inaudio'][id] is None:
            inputData, sr = librosa.load(dataset['innames'][id], sr=None)
            outputData, sr = librosa.load(dataset['outnames'][id], sr=None)

            shape = np.shape(inputData)

            dataset['inaudio'][id] = np.float32(inputData)
            dataset['outaudio'][id] = np.float32(outputData)

    return dataset


class AudioDataset(data.Dataset):
    """
    Audio sample reader.
    """

    def __init__(self, data_type):
        dataset = load_data_list(setname=data_type)
        self.dataset = load_data(dataset)

        self.file_names = dataset['innames']

    def __getitem__(self, idx):
        mixed = torch.from_numpy(self.dataset['inaudio'][idx]).type(torch.FloatTensor)
        clean = torch.from_numpy(self.dataset['outaudio'][idx]).type(torch.FloatTensor)

        return mixed, clean

    def __len__(self):
        return len(self.file_names)

    def zero_pad_concat(self, inputs):
        max_t = max(inp.shape[0] for inp in inputs)
        shape = (len(inputs), max_t)
        input_mat = np.zeros(shape, dtype=np.float32)
        for e, inp in enumerate(inputs):
            input_mat[e, :inp.shape[0]] = inp
        return input_mat

    def collate(self, inputs):
        mixeds, cleans = zip(*inputs)
        seq_lens = torch.IntTensor([i.shape[0] for i in mixeds])

        x = torch.FloatTensor(self.zero_pad_concat(mixeds))
        y = torch.FloatTensor(self.zero_pad_concat(cleans))

        batch = [x, y, seq_lens]
        return batch
