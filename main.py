from load_data import load_data
import os
import numpy as np
import torch
from helper_functions import one_hot, select_channels, train_test_model, cross_validation_1_layer


def main():
    # if not os.path.isfile("./data/filtered_data/signals.npy") and not os.path.isfile("./data/filtered_data/targets.npy"):
    #     load_data()

    channel_list = ['C1','C2']
    K = 10
    lr = 1e-5
    minibatch = False
    batch_size = 64
    num_epochs = 2000
    if minibatch:
        run_name = f'_lr_{lr}_bs_{batch_size}'
    else:
        run_name = f'_lr_{lr}_nobs'

    file = open("./results/log"+run_name+".txt", "w")

    X = np.load("./data/filtered_data/signals.npy")
    y_pre = np.load("./data/filtered_data/targets.npy")
    y, encoding = one_hot(y_pre)

    electrodes = np.load("./data/filtered_data/electrodes.npy")

    X = torch.from_numpy(X).float()
    X = select_channels(channel_list, X, electrodes)
    y = torch.from_numpy(y)

    train_acc, test_acc, losses, train_conf, test_conf = cross_validation_1_layer(X, y, K, lr, batch_size, num_epochs, minibatch=minibatch, output_file=file)

    np.save("./results/train_accuracies" + run_name, train_acc)
    np.save("./results/test_accuracies" + run_name, test_acc)
    np.save("./results/losses" + run_name, losses)
    np.save("./results/train_confusion" + run_name, train_conf)
    np.save("./results/test_confusion" + run_name, test_conf)

    file.close()

if __name__ == "__main__":
    main()
