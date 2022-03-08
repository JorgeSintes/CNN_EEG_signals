from load_data import load_data
import os
import numpy as np
import torch
from helper_functions import one_hot, select_channels, train_test_model, cross_validation_1_layer


def main():
    # if not os.path.isfile("./data/filtered_data/signals.npy") and not os.path.isfile("./data/filtered_data/targets.npy"):
    #     load_data()

    channel_list = [('Fc1','Fc2'),('Fc3','Fc4'),('Fc5','Fc6'),('C1','C2'),('C3','C4'),
                    ('C5','C6'),('Cp1','Cp2'),('Cp3','Cp4'),('Cp5','Cp6')]
    K = 10
    lr = 1e-3
    wd = 0
    minibatch = False
    batch_size = 64
    num_epochs = 2000
    nb_subs = 2

    if len(channel_list) == 9:
        channel_name = 'all'
    elif len(channel_list) == 1:
        channel_name = channel_list[0]
    else:
        channel_name = channel_list

    if minibatch:
        run_name = f'_lr_{lr}_bs_{batch_size}_ch_{channel_name}'
    else:
        run_name = f'_lr_{lr}_nobs_ch_{channel_name}'

    file = open("./results/log"+run_name+".txt", "w")

    X = np.load("./data/filtered_data/signals_ordered.npy")[:nb_subs, :, :, :]
    y_pre = np.load("./data/filtered_data/targets_ordered.npy")[:nb_subs, :]

    electrodes = np.load("./data/filtered_data/electrodes.npy")

    X = torch.from_numpy(X).float()

    train_acc, test_acc, losses, train_conf, test_conf = cross_validation_1_layer(X, y_pre, channel_list, electrodes, K, lr, wd, batch_size, num_epochs, minibatch=minibatch, output_file=file)

    np.save("./results/train_accuracies" + run_name, train_acc)
    np.save("./results/test_accuracies" + run_name, test_acc)
    np.save("./results/losses" + run_name, losses)
    np.save("./results/train_confusion" + run_name, train_conf)
    np.save("./results/test_confusion" + run_name, test_conf)

    file.close()

if __name__ == "__main__":
    main()
