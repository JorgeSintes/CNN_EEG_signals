from load_data import load_data
import os
import numpy as np
import torch
from helper_functions import cross_validation_1_layer, plot_ensemble


def main():
    # if not os.path.isfile("./data/filtered_data/signals.npy") and not os.path.isfile("./data/filtered_data/targets.npy"):
    #     load_data()

    K = 5
    lr = 1e-5
    wd = 0
    batch_size = 16
    num_epochs = 100
    nb_models = 10
    nb_subs = 10
    w_init_params = (0, 2)      # mean and std of initialized weights of models in Ensemble
    classes = ["L", "R", "0"]   # selected classes - possible classes: "L", "R", "LR", "F", "0"

    if batch_size:
        run_name = f'_lr_{lr}_bs_{batch_size}_classes_{len(classes)}_models_{nb_models}_w_init_{w_init_params[1]}'
    else:
        run_name = f'_lr_{lr}_nobs_classes_{len(classes)}_models_{nb_models}_w_init_{w_init_params[1]}'

    file = open("./results/log"+run_name+".txt", "w")

    X = np.load("./data/filtered_data/signals_ordered_6s_all_tags.npy")[:nb_subs, :, :, :]
    y_pre = np.load("./data/filtered_data/targets_ordered_6s_all_tags.npy")[:nb_subs, :]

    mask = np.isin(y_pre[0], classes)   # only works if class repartion is the same for all subjects

    y_pre = y_pre[:, mask]
    X = X[:, mask, :, :]

    X = torch.from_numpy(X).float()

    # train_losses, test_losses, train_acc, test_acc, train_conf, test_conf = cross_validation_1_layer(X, y_pre, K, nb_models, lr, wd, batch_size, num_epochs, nb_classes=len(classes), output_file=file, run_name=run_name, w_init_params=w_init_params)

    # np.save("./results/train_losses" + run_name, train_losses)
    # np.save("./results/test_losses" + run_name, test_losses)
    # np.save("./results/train_accuracies" + run_name, train_acc)
    # np.save("./results/test_accuracies" + run_name, test_acc)
    # np.save("./results/train_confusion" + run_name, train_conf)
    # np.save("./results/test_confusion" + run_name, test_conf)

    plot_ensemble(X, y_pre, K, batch_size, nb_models, len(classes), 2, f"_lr_{lr}_bs_{batch_size}_classes_{len(classes)}_models_{nb_models}_w_init_{w_init_params[1]}")

    file.close()

if __name__ == "__main__":
    main()
