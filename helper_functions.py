import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from paper_network import Ensemble
import itertools

sbn.set_style('darkgrid')

def one_hot(array):
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    onehot = onehot.reshape(array.shape[0], -1, len(unique))
    return onehot, unique


def cross_validation_1_layer(X, y_pre, K, nb_models, lr=1e-5, wd=0, batch_size=64, num_epochs=2000, nb_classes=4, output_file=None, run_name="", w_init_params=(0,1), weights_folder=""):
    CV = StratifiedKFold(n_splits=K, shuffle=True, random_state=12)

    train_acc = np.zeros((K, num_epochs))
    test_acc = np.zeros((K, num_epochs))
    train_losses = np.zeros((K, nb_models, num_epochs))
    test_losses = np.zeros((K, nb_models, num_epochs))
    train_conf = np.zeros((K, nb_classes, nb_classes))
    test_conf = np.zeros((K, nb_classes, nb_classes))

    y, encoding = one_hot(y_pre)
    y = torch.from_numpy(y)

    # Making the CV dependent on the shape of X which differs depending on 'separated'
    split_gen = CV.split(np.zeros((X.shape[1], 1)), y_pre[0,:])

    for i, (train_index, test_index) in enumerate(split_gen):

        # Taking train and test slices over subjects 
        X_train, y_train = X[:, train_index, :, :], y[:, train_index, :]
        X_test, y_test = X[:, test_index, :, :], y[:, test_index, :]

        # Reshaping arrays to have all observations in first dimension
        X_train = X_train.reshape(-1, X.shape[2], X.shape[3])
        y_train = y_train.reshape(-1, nb_classes)

        X_test = X_test.reshape(-1, X.shape[2], X.shape[3])
        y_test = y_test.reshape(-1, nb_classes)

        # Standardising
        mu = torch.mean(X_train, dim=(0,2)).reshape(1,-1,1)
        sigma = torch.std(X_train, dim=(0,2)).reshape(1,-1,1)
        X_train = (X_train - mu) / sigma
        X_test  = (X_test - mu) / sigma

        model = Ensemble(nb_models, nb_classes, signal_length=X.shape[3], output_file=output_file, run_name=run_name, transfer_to_device=True, k=i+1, w_init_params=w_init_params)

        train_losses[i, :, :], test_losses[i, :, :], train_acc[i, :], test_acc[i, :], train_conf[i, :, :], test_conf[i, :, :] = model.train_test_on_the_fly(X_train, y_train, X_test, y_test, num_epochs, lr=lr, wd=wd, batch_size=batch_size, verbose=True)
        model.save_weights(weights_folder)

    return train_losses, test_losses, train_acc, test_acc, train_conf, test_conf


def get_acc_fold(X, y, train_index, test_index, batch_size, nb_classes, nb_models, fold, run_name, swa_params):
    # Taking train and test slices over subjects 
    X_train, y_train = X[:, train_index, :, :], y[:, train_index, :]
    X_test, y_test = X[:, test_index, :, :], y[:, test_index, :]

    # Reshaping arrays to have all observations in first dimension
    X_train = X_train.reshape(-1, X.shape[2], X.shape[3])
    y_train = y_train.reshape(-1, nb_classes)

    X_test = X_test.reshape(-1, X.shape[2], X.shape[3])
    y_test = y_test.reshape(-1, nb_classes)

    # Standardising
    mu = torch.mean(X_train, dim=(0,2)).reshape(1,-1,1)
    sigma = torch.std(X_train, dim=(0,2)).reshape(1,-1,1)
    X_train = (X_train - mu) / sigma
    X_test  = (X_test - mu) / sigma

    model = Ensemble(nb_models, nb_classes, k=fold, run_name=run_name, transfer_to_device=True)
    model.load_weights()
    accuracies = []
    single_accuracies = []
    accuracies_swa = []
    single_accuracies_swa = []
    log_pred_densities = []
    log_pred_densities_swa = []

    for i in range(nb_models):
        metrics = model.test(X_test, y_test, batch_size, models_used=i+1)
        accuracies.append(metrics["acc"])
        log_pred_densities.append(metrics["log_pred_dens"])

        metrics = model.test(X_test, y_test, batch_size, models_used=[i])
        single_accuracies.append(metrics["acc"])

    if swa_params:
        model.do_the_swa(X_train, y_train, swa_params["num_epochs"], swa_params["lr"], swa_params["K"], c=swa_params["c"], batch_size=batch_size)

        for i in range(nb_models):
            metrics = model.test(X_test, y_test, batch_size, models_used=i+1)
            accuracies_swa.append(metrics["acc"])
            log_pred_densities_swa.append(metrics["log_pred_dens"])

            metrics = model.test(X_test, y_test, batch_size, models_used=[i])
            single_accuracies_swa.append(metrics["acc"])

    return {"acc": accuracies, "single_acc": single_accuracies, "acc_swa": accuracies_swa, "single_acc_swa": single_accuracies_swa, "log_pred_dens": log_pred_densities, "log_pred_dens_swa": log_pred_densities_swa}


def plot_ensemble(X, y_pre, K, batch_size, nb_models, nb_classes, fold, run_name, swa_params=None, alpha=0.5):

    CV = StratifiedKFold(n_splits=K, shuffle=True, random_state=12)
    split_gen = CV.split(np.zeros((X.shape[1], 1)), y_pre[0,:])

    y, encoding = one_hot(y_pre)
    y = torch.from_numpy(y)

    train_index, test_index = next(itertools.islice(split_gen, fold-1, None))

    #for _ in range(fold):
    #    train_index, test_index = next(split_gen)

    metrics = get_acc_fold(X, y, train_index, test_index, batch_size, nb_classes, nb_models, fold, run_name, swa_params)

    accuracies = metrics["acc"]
    single_accuracies = metrics["single_acc"]
    accuracies_swa = metrics["acc_swa"]
    single_accuracies_swa = metrics["single_acc_swa"]
    log_pred_densities = metrics["log_pred_dens"]
    log_pred_densities_swa = metrics["log_pred_dens_swa"]

    fig, ax = plt.subplots(1,2, figsize=(20,10))
    ax[0].plot(list(range(1, nb_models + 1)), accuracies, 'b', label="Test accuracies")
    ax[1].plot(list(range(1, nb_models + 1)), log_pred_densities, 'b', label="Test LPD")

    for single_acc in single_accuracies:
        ax[0].plot(list(range(1, nb_models + 1)), [single_acc]*nb_models, 'g', alpha=alpha)

    if swa_params:
        ax[0].plot(list(range(1, nb_models + 1)), accuracies_swa, 'r', label="Test accuracies after SWA")
        ax[1].plot(list(range(1, nb_models + 1)), log_pred_densities_swa, 'r', label="Test LPD after SWA")

        for single_acc in single_accuracies_swa:
            ax[0].plot(list(range(1, nb_models + 1)), [single_acc]*nb_models, 'orange', alpha=alpha)

    ax[0].set(xlabel="No. of models", ylabel="Accuracy")
    ax[1].set(xlabel="No. of models", ylabel="Log pred dens")
    ax[0].legend()
    ax[1].legend()
    fig.suptitle(f"Ensemble - Fold: {fold}, classes: {nb_classes}")
    fig.savefig(f"./figures/"+run_name[1:]+f"_{fold}_fold.pdf")


def plot_ensemble_all(X, y_pre, K, batch_size, nb_models, nb_classes, run_name, swa_params=None, alpha=0.5):

    CV = StratifiedKFold(n_splits=K, shuffle=True, random_state=12)
    split_gen = CV.split(np.zeros((X.shape[1], 1)), y_pre[0,:])

    y, encoding = one_hot(y_pre)
    y = torch.from_numpy(y)

    accuracies_ens_all_folds = []
    accuracies_swa_all_folds = []
    log_pred_dens_ens_all_folds = []
    log_pred_dens_swa_all_folds = []
    accuracies_models_all_folds = []
    accuracies_models_all_folds_swa = []

    for fold, (train_index, test_index) in enumerate(split_gen):

        metrics = get_acc_fold(X, y, train_index, test_index, batch_size, nb_classes,
                                                       nb_models, fold+1, run_name, swa_params)

        accuracies_ens_all_folds.append(metrics["acc"])
        accuracies_swa_all_folds.append(metrics["acc_swa"])
        
        log_pred_dens_ens_all_folds.append(metrics["log_pred_dens"])
        log_pred_dens_swa_all_folds.append(metrics["log_pred_dens_swa"])

        accuracies_models_all_folds.append(metrics["single_acc"])        
        accuracies_models_all_folds_swa.append(metrics["single_acc_swa"]) 
        
    accuracies_ens_all_folds = np.asarray(accuracies_ens_all_folds)
    accuracies_swa_all_folds = np.asarray(accuracies_swa_all_folds)
    log_pred_dens_ens_all_folds = np.asarray(log_pred_dens_ens_all_folds)
    log_pred_dens_swa_all_folds = np.asarray(log_pred_dens_swa_all_folds)
    accuracies_models_all_folds = np.asarray(accuracies_models_all_folds)
    accuracies_models_all_folds_swa = np.asarray(accuracies_models_all_folds_swa)

    avg_accs_ens = np.mean(accuracies_ens_all_folds, axis=0)
    avg_accs_swa = np.mean(accuracies_swa_all_folds, axis=0)
    avg_log_pred_dens_ens = np.mean(log_pred_dens_ens_all_folds, axis=0)
    avg_log_pred_dens_swa = np.mean(log_pred_dens_swa_all_folds, axis=0)
    avg_acc_models = np.mean(accuracies_models_all_folds, axis=0)
    avg_acc_models_swa = np.mean(accuracies_models_all_folds_swa, axis=0)

    ste_accs_ens = np.std(accuracies_ens_all_folds, axis=0)/np.sqrt(5)
    ste_accs_swa = np.std(accuracies_swa_all_folds, axis=0)/np.sqrt(5)
    ste_log_pred_dens_ens = np.std(log_pred_dens_ens_all_folds, axis=0)/np.sqrt(5)
    ste_log_pred_dens_swa = np.std(log_pred_dens_swa_all_folds, axis=0)/np.sqrt(5)

    fig, ax = plt.subplots(1,2, figsize=(20,10))

    # ax[0].plot(list(range(1, nb_models + 1)), avg_accs_ens, c='b', label='ensemble')
    ax[0].errorbar(list(range(1, nb_models + 1)), avg_accs_ens, yerr=ste_accs_ens, c='C0', label="ensemble", alpha=alpha)

    # ax[0].plot(list(range(1, nb_models + 1)), avg_accs_swa, c='g', label='swag')
    ax[0].errorbar(list(range(1, nb_models + 1)), avg_accs_swa, yerr=ste_accs_swa, c='C1', label="swag", alpha=alpha)
    
    ax[0].plot(list(range(1, nb_models + 1)), avg_acc_models, c='C0', label='single models', alpha=alpha)
    ax[0].plot(list(range(1, nb_models + 1)), avg_acc_models_swa, c='C1', label='single models swa', alpha=alpha)
    ax[0].axline((1,np.mean(avg_acc_models)),(10,np.mean(avg_acc_models)), c='C0', linestyle='--', label='avg single models', alpha=alpha)
    ax[0].axline((1,np.mean(avg_acc_models_swa)),(10,np.mean(avg_acc_models_swa)), c='C1', linestyle='--', label='avg single models swa', alpha=alpha)

    ax[0].set(xlabel="No. of models", ylabel="Average accuracy", title='Average accuracy of models on test data across folds with errorbars')
    ax[0].legend()

    # ax[1].plot(list(range(1, nb_models + 1)), avg_log_pred_dens_ens, c='b', label='ensemble')
    ax[1].errorbar(list(range(1, nb_models + 1)), avg_log_pred_dens_ens, yerr=ste_log_pred_dens_ens, c='C0', label="ensemble", alpha=alpha)

    # ax[1].plot(list(range(1, nb_models + 1)), avg_log_pred_dens_swa, c='g', label='swag')
    ax[1].errorbar(list(range(1, nb_models + 1)), avg_log_pred_dens_swa, yerr=ste_log_pred_dens_swa, c='C1', label="swag", alpha=alpha)

    ax[1].set(xlabel="No. of models", ylabel="Average log PD", title='Average log pred dens of models on test data across folds with errorbars')
    ax[1].legend()
    fig.savefig(f"./figures/"+run_name[1:]+f"_avg_acc_fols_w-errorbars.pdf", bbox_inches='tight', format='pdf')

