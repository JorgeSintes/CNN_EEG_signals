# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 21:51:19 2022

@author: cleml
"""

import numpy as np
import torch

def one_hot(array):
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot

def train_test_model(
    model,
    criterion,
    optimizer,
    X_train,
    y_train,
    X_test,
    y_test,
    batch_size=100,
    num_epochs=10,
    transfer_to_device=True,
):
    if transfer_to_device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device}")
        model = model.to(device)
        criterion = criterion.to(device)
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_test = X_test.to(device)
        y_test = y_test.to(device)

    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    train_nbatches = train_size // batch_size
    test_nbatches = train_size // batch_size

    train_acc, test_acc = [], []
    losses = []
    for epoch in range(num_epochs):

        ### Train
        acum_loss = 0
        model.train()
        for i in range(train_nbatches):
            X_batch = X_train[i * batch_size : (i + 1) * batch_size]
            y_batch = y_train[i * batch_size : (i + 1) * batch_size].long()
            out = model(X_batch)

            # Compute Loss and gradients
            batch_loss = criterion(out, y_batch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            acum_loss += batch_loss

        losses.append(acum_loss / batch_size)

        ### Evaluate
        train_preds, test_preds = [], []
        model.eval()

        # First on training set
        for i in range(train_nbatches):
            X_batch = X_train[i * batch_size : (i + 1) * batch_size]
            y_batch = y_train[i * batch_size : (i + 1) * batch_size].long()
            out = model(X_batch)
            preds = torch.max(out, 1)[1].to("cpu")
            train_preds += list(preds.data.numpy())

        # Then on test
        for i in range(test_nbatches):
            X_batch = X_test[i * batch_size : (i + 1) * batch_size]
            y_batch = y_test[i * batch_size : (i + 1) * batch_size]
            out = model(X_batch)
            preds = torch.max(out, 1)[1].to("cpu")
            test_preds += list(preds.data.numpy())

        # Compute accuracies
        train_acc_cur = accuracy_score(y_train.to("cpu"), train_preds)
        test_acc_cur = accuracy_score(y_test.to("cpu"), test_preds)

        train_acc.append(train_acc_cur)
        test_acc.append(test_acc_cur)

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch}: Train Loss {losses[-1]:.4f}, Train Accur {train_acc_cur:.4f}, Test Accur {test_acc_cur:.4f}"
            )

    return train_acc, test_acc, losses