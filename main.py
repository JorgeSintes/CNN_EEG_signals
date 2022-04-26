from load_data import load_data
import os
import numpy as np
import torch
from helper_functions import one_hot, select_channels, train_test_model, cross_validation_1_layer


def main():
    # if not os.path.isfile("./data/filtered_data/signals.npy") and not os.path.isfile("./data/filtered_data/targets.npy"):
    #     load_data()

    K = 5
    lr = 1e-5
    wd = 0
    minibatch = True
    batch_size = 16
    num_epochs = 200
    nb_subs = -1
    classes = ["L", "R", "0"]           # selected classes - possible classes: "L", "R", "LR", "F", "0"

    if minibatch:
        run_name = f'_new_net_lr_{lr}_bs_{batch_size}_subs_{nb_subs}_epochs_{num_epochs}'
    else:
        run_name = f'_new_net_lr_{lr}_nobs_subs_{nb_subs}_epochs_{num_epochs}'

    file = open("./results/log"+run_name+".txt", "w")

    X = np.load("./data/filtered_data/signals_ordered_6s_all_tags.npy")[:nb_subs, :, :, :]
    y_pre = np.load("./data/filtered_data/targets_ordered_6s_all_tags.npy")[:nb_subs, :]

    mask = np.isin(y_pre[0], classes)   # only works if class repartion is the same for all subjects

    y_pre = y_pre[:, mask]
    X = X[:, mask, :, :]

    electrodes = np.load("./data/filtered_data/electrodes.npy")

    X = torch.from_numpy(X).float()

    train_acc, test_acc, losses, train_conf, test_conf = cross_validation_1_layer(X, y_pre, electrodes, K, lr, wd, batch_size, num_epochs, nb_classes=len(classes), minibatch=minibatch, output_file=file)

    np.save("./results/train_accuracies" + run_name, train_acc)
    np.save("./results/test_accuracies" + run_name, test_acc)
    np.save("./results/losses" + run_name, losses)
    np.save("./results/train_confusion" + run_name, train_conf)
    np.save("./results/test_confusion" + run_name, test_conf)

    file.close()

if __name__ == "__main__":
    main()
