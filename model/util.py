import numpy as np
import torchaudio.functional as F
import torchaudio
import torch
from torch import nn
from pathlib import Path
# import sys
FAKE_DIR = "data/KAGGLE/AUDIO/FAKE"
REAL_DIR = "data/KAGGLE/AUDIO/REAL"

class HyperPrams: # Prameters Class
    def __init__(self, prams_dict:dict) -> None:
        for i in prams_dict.keys():
            self.__setattr__(i, prams_dict[i])

def load(dir:str, pram:HyperPrams):
    paths = [str(path) for path in Path(dir).glob("*.wav")]
    audios = []
    for path in paths:
        try:
            audio, sr = torchaudio.load(path)

            if sr != pram.SR:
                F.resample(audio, sr, pram.SR)

            if(audio.shape[0] == 2):
                audio = torch.mean(audio, 0)

            length = audio.shape[-1]
            # print(audio.shape,"+++")


            for i in range(0, length - (pram.SR*pram.SPLIT_S), (pram.SR*pram.SPLIT_S)):
                res = audio[i : i+(pram.SR*pram.SPLIT_S)]
                if res.shape[0] == (pram.SR*pram.SPLIT_S):
                    audios.append(res.numpy())
            
        except :
            print("oh.. Noooooo!!!\n\t err! err!")
            # sys.exit()
    # print(len(audios))
    return np.array(audios)

class Process_mel:
    def __init__(self, pram:HyperPrams) -> None:
        self.transform = nn.Sequential( torchaudio.transforms.MelSpectrogram(pram.SR, n_fft=pram.NFFT, win_length=pram.WINLEN, hop_length=pram.HOP, n_mels=pram.NMEL),\
                              torchaudio.transforms.AmplitudeToDB())
    def __call__(self, arr:torch.Tensor):
        return self.transform(arr)