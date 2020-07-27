#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils import data
import numpy as np
import os
from sklearn.model_selection import train_test_split

from .lorentz import to_p4

class MyDataset(data.Dataset):
    def __init__(self, X, target, r, c):
        self.X = X #torch.from_numpy(X).half()
        self.target = torch.from_numpy(target)       
        self.r = r
        self.c = c

    def _to_img(self, p4, r, c):
        """
        project the evt_p4 on to (theta, phi) plane with E on pixels
        """
        img = np.zeros((1, r, c))
        for p in p4:
            if p.E > 0:
                [n_r, n_c] = [int(p.theta/(np.pi/c)), int((p.phi+np.pi)/(2*np.pi/r))]
                if n_r<r and n_c<c and n_r>=0 and n_c>=0:
                    img[0, n_r, n_c] += p.E
        return img

    def __getitem__(self, index):
        evt_p4 = self.X[index]
        img = torch.from_numpy(self._to_img(evt_p4, self.r, self.c))
        y = self.target[index]
        return img, y
        
    def __len__(self):
        return len(self.X)

def get_data(sig, bg, r, c, batch_size, test_size=0.2, use_gpu=True):
    ## for BCE loss
    target_sig = np.ones((len(sig), 1), dtype=np.float32)
    target_bg = np.zeros((len(bg), 1), dtype=np.float32)
    
    X = np.concatenate((sig, bg), axis=0)
    X = to_p4(X)
    del sig, bg
    target = np.concatenate((target_sig, target_bg), axis=0)

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X, target = X[idx], target[idx]
    X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=test_size)
    del X
    trainset = MyDataset(X_train, target_train, r, c)
    testset = MyDataset(X_test, target_test, r, c)
    
    pin_memory = True if use_gpu else False
    loaders = {
        'train': data.DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=6),
        'val': data.DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=6)
    }
    return loaders




