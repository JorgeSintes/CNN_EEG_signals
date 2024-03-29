from load_data import load_data
import os
import json
import time
import numpy as np
import torch
from helper_functions import cross_validation_1_layer, plot_ensemble, plot_ensemble_all, do_the_swag, get_data

def main(plot=False):
    # if not os.path.isfile("./data/filtered_data/signals.npy") and not os.path.isfile("./data/filtered_data/targets.npy"):
    #     load_data()

    K = 5                       # Number of cross-validation folds 
    lr = 1e-5                   # learning rate
    wd = 0                      # weight decay of optimizer
    batch_size = 16
    num_epochs = 100
    nb_models = 10
    nb_subs = 100                   # number of subjscts
    w_init_params = (0, False)      # mean and std of initialized weights of models in Ensemble
    classes = ["L", "R", "0"]       # selected classes - possible classes: "L", "R", "LR", "F", "0"

    info_dict = {
                'K': K,
                'lr': lr,
                'wd': wd,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'nb_models': nb_models,
                'nb_subs': nb_subs,
                'w_init_params': w_init_params,
                'classes': classes
            }

    swag_params = {
            "num_epochs": 5,
            "lr": 1e-5,
            "K": 3,                     # Number of model weights to construct covariance matrix of swag from, see paper
            "c": 1,                     # every c-th iteration of SGD for swa counts for computing mean and covariance matrix of swa
            "swa": True,                # Do swag if true
            "swag": True,               # Do swag if true
            "S": 30                     # No. of samples in swag test
            }

    if batch_size:
        run_name = f'_lr_{lr}_bs_{batch_size}_classes_{len(classes)}_models_{nb_models}_w_init_{w_init_params[1]}'
    else:
        run_name = f'_lr_{lr}_nobs_classes_{len(classes)}_models_{nb_models}_w_init_{w_init_params[1]}'

    if not plot:
        date = time.localtime()
        date_str = f"{date.tm_year}-{date.tm_mon}-{date.tm_mday}_{date.tm_hour}-{date.tm_min}-{date.tm_sec}"
        folder_name = date_str + run_name
        path = "./results/" + folder_name + "/"
        os.mkdir(path)

        weights_path = "./models/" + folder_name + "/"
        os.mkdir(weights_path)

        info_file = open(path + "info.json", "w")
        json.dump(info_dict, info_file)
        info_file.close()

        file = open(path + "log.txt", "w")

    X, y_pre = get_data(nb_subs, classes)

    if not plot:
        metrics = cross_validation_1_layer(X, y_pre, K, nb_models, lr, wd, batch_size, num_epochs, nb_classes=len(classes), output_file=file, run_name=run_name, w_init_params=w_init_params, weights_folder=folder_name)

        for name, element in zip(metrics.keys(), metrics.values()):
            np.save(path + name, element)

        file.close()

    if plot:
        # run_name = f"_lr_{lr}_bs_{batch_size}_classes_{len(classes)}_models_{nb_models}_w_init_{w_init_params[1]}"
        # do_the_swag(X, y_pre, K, batch_size, nb_models, len(classes), run_name, swag_params=swag_params)
        for k in range(1,K+1):
            plot_ensemble(X, y_pre, K, batch_size, nb_models, len(classes), k, run_name, swag_params=swag_params)
        # plot_ensemble_all(X, y_pre, K, batch_size, nb_models, len(classes), run_name, swag_params=swag_params)

if __name__ == "__main__":
    main(plot=True)
