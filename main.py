# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 16:59:15 2022

@author: cleml
"""

from load_data import load_data
from paper_network import Network
import os
import numpy as np
import torch
from helper_functions import one_hot

if not os.path.isfile("./data/filtered_data/signals.npy") and not os.path.isfile("./data/filtered_data/targets.npy"):

    load_data()
    
X = np.load("./data/filtered_data/signals.npy")
y = np.load("./data/filtered_data/targets.npy")
y = one_hot(y)

X = torch.from_numpy(X).float()
y = torch.from_numpy(y)

net = Network()

pred = net.forward(X[0,:2,:])

print(pred)