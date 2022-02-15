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
from helper_functions import one_hot, select_channels, train_test_model
from sklearn.model_selection import StratifiedKFold

# if not os.path.isfile("./data/filtered_data/signals.npy") and not os.path.isfile("./data/filtered_data/targets.npy"):
#     load_data()

channel_list = ['C1','C2']
lr = 0.005
batch_size = 64
num_epochs = 20

X = np.load("./data/filtered_data/signals.npy")
y_pre = np.load("./data/filtered_data/targets.npy")
y, encoding = one_hot(y_pre)

electrodes = np.load("./data/filtered_data/electrodes.npy")

X = torch.from_numpy(X).float()
X = select_channels(channel_list, X, electrodes)
y = torch.from_numpy(y)

skf = StratifiedKFold(n_splits=5)

for train_index, test_index in skf.split(X,y_pre):

    X_train, y_train = X[train_index,:,:], y[train_index,:]
    X_test, y_test = X[test_index,:,:], y[test_index,:]
    break

model = Network()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


train_test_model(model, criterion, optimizer, X_train, y_train, X_test, y_test, batch_size, num_epochs)
