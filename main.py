from load_data import load_data
import os
import numpy as np
import torch
from helper_functions import one_hot, select_channels, train_test_model, cross_validation_1_layer


def main():
    # if not os.path.isfile("./data/filtered_data/signals.npy") and not os.path.isfile("./data/filtered_data/targets.npy"):
    #     load_data()

    K = 10
    lr = 1e-4
    wd = 0
    minibatch = True
    batch_size = 32
    num_epochs = 2000
    nb_subs = 10

    if minibatch:
        run_name = f'new_net_lr_{lr}_bs_{batch_size}'
    else:
        run_name = f'_lr_{lr}_nobs_ch_{channel_name}'

    file = open("./results/log"+run_name+".txt", "w")

    X = np.load("./data/filtered_data/signals_ordered.npy")[:nb_subs, :, :, :]
    y_pre = np.load("./data/filtered_data/targets_ordered.npy")[:nb_subs, :]

    electrodes = np.load("./data/filtered_data/electrodes.npy")

    X = torch.from_numpy(X).float()

    train_acc, test_acc, losses, train_conf, test_conf = cross_validation_1_layer(X, y_pre, electrodes, K, lr, wd, batch_size, num_epochs, minibatch=minibatch, output_file=file)

    np.save("./results/train_accuracies" + run_name, train_acc)
    np.save("./results/test_accuracies" + run_name, test_acc)
    np.save("./results/losses" + run_name, losses)
    np.save("./results/train_confusion" + run_name, train_conf)
    np.save("./results/test_confusion" + run_name, test_conf)

    file.close()

if __name__ == "__main__":
    main()
