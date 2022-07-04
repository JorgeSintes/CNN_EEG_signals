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


def cross_validation_1_layer(X, y_pre, K, nb_models, lr=1e-5, wd=0, batch_size=64, num_epochs=2000, nb_classes=4, output_file=None, run_name="", w_init_params=(0,False), weights_folder=""):
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


# def plot_ensemble_all(X, y_pre, K, batch_size, nb_models, nb_classes, run_name, swag_params=None, alpha=0.5):
def do_the_swag(X, y_pre, K, batch_size, nb_models, nb_classes, run_name, swag_params, swag=True):
    CV = StratifiedKFold(n_splits=K, shuffle=True, random_state=12)

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

        model = Ensemble(nb_models, nb_classes, k=i+1, run_name=run_name, transfer_to_device=True)
        model.load_weights()
        model.train_swag(X_train, y_train, num_epochs=swag_params["num_epochs"], lr=swag_params["lr"], K=swag_params["K"], c=swag_params["c"], batch_size=batch_size, swag=swag_params["swag"])
        model.save_swag_results()


def get_acc_fold(X, y, train_index, test_index, batch_size, nb_classes, nb_models, fold, run_name, swag_params):
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
    accuracies_swa = []
    accuracies_swag = []

    single_accuracies = []
    single_accuracies_swa = []
    single_accuracies_swag = []

    single_log_preds = []
    single_log_preds_swa = []
    single_log_preds_swag = []

    log_pred_densities = []
    log_pred_densities_swa = []
    log_pred_densities_swag = []

    for i in range(nb_models):
        metrics = model.test(X_test, y_test, model.inference, batch_size, models_used=i+1)
        accuracies.append(metrics["acc"])
        log_pred_densities.append(metrics["log_pred_dens"])

        metrics = model.test(X_test, y_test, model.inference, batch_size, models_used=[i])
        single_accuracies.append(metrics["acc"])
        single_log_preds.append(metrics["log_pred_dens"])

    if swag_params:

        model.load_swag_results()

        # SWA PART

        if swag_params['swa']:
        # model.train_swag(X_train, y_train, swag_params["num_epochs"], swag_params["lr"], swag_params["K"], c=swag_params["c"], batch_size=batch_size, swag=False)


            for i in range(nb_models):
                metrics = model.test(X_test, y_test, model.inference, batch_size, models_used=i+1)
                accuracies_swa.append(metrics["acc"])
                log_pred_densities_swa.append(metrics["log_pred_dens"])

                metrics = model.test(X_test, y_test, model.inference, batch_size, models_used=[i])
                single_accuracies_swa.append(metrics["acc"])
                single_log_preds_swa.append(metrics["log_pred_dens"])


        # SWAG PART

        if swag_params['swag']:


            for i in range(nb_models):
                metrics = model.test(X_test, y_test, model.swag_inference, batch_size, models_used=i+1)
                accuracies_swag.append(metrics["acc"])
                log_pred_densities_swag.append(metrics["log_pred_dens"])

                metrics = model.test(X_test, y_test, model.swag_inference, batch_size, models_used=[i])
                single_accuracies_swag.append(metrics["acc"])
                single_log_preds_swag.append(metrics["log_pred_dens"])

    return {"acc": accuracies, "acc_swa": accuracies_swa, "acc_swag": accuracies_swag,
            "single_acc": single_accuracies, "single_acc_swa": single_accuracies_swa, "single_acc_swag": single_accuracies_swag,
            "single_log_preds": single_log_preds, "single_log_preds_swa": single_log_preds_swa, "single_log_preds_swag": single_log_preds_swag,
            "log_pred_dens": log_pred_densities, "log_pred_dens_swa": log_pred_densities_swa, "log_pred_dens_swag": log_pred_densities_swag}


def plot_ensemble(X, y_pre, K, batch_size, nb_models, nb_classes, fold, run_name, swag_params=None, alpha=0.5):

    CV = StratifiedKFold(n_splits=K, shuffle=True, random_state=12)
    split_gen = CV.split(np.zeros((X.shape[1], 1)), y_pre[0,:])

    y, encoding = one_hot(y_pre)
    y = torch.from_numpy(y)

    train_index, test_index = next(itertools.islice(split_gen, fold-1, None))

    #for _ in range(fold):
    #    train_index, test_index = next(split_gen)

    metrics = get_acc_fold(X, y, train_index, test_index, batch_size, nb_classes, nb_models, fold, run_name, swag_params)

    accuracies = metrics["acc"]
    single_accuracies = metrics["single_acc"]
    single_log_preds = metrics["single_log_preds"]
    accuracies_swa = metrics["acc_swa"]
    single_accuracies_swa = metrics["single_acc_swa"]
    single_log_preds_swa = metrics["single_log_preds_swa"]
    log_pred_densities = metrics["log_pred_dens"]
    log_pred_densities_swa = metrics["log_pred_dens_swa"]

    fig, ax = plt.subplots(1,2, figsize=(20,10))
    ax[0].plot(list(range(1, nb_models + 1)), accuracies, 'b', label="Test accuracies")
    ax[1].plot(list(range(1, nb_models + 1)), log_pred_densities, 'b', label="Test LPD")

    for single_acc in single_accuracies:
        ax[0].plot(list(range(1, nb_models + 1)), [single_acc]*nb_models, 'g', alpha=alpha)

    if swag_params:
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


def plot_ensemble_all(X, y_pre, K, batch_size, nb_models, nb_classes, run_name, swag_params=None, alpha=0.5):

    CV = StratifiedKFold(n_splits=K, shuffle=True, random_state=12)
    split_gen = CV.split(np.zeros((X.shape[1], 1)), y_pre[0,:])

    y, encoding = one_hot(y_pre)
    y = torch.from_numpy(y)

    acc_ens = np.zeros((K, nb_models))
    acc_swa = np.zeros((K, nb_models))
    acc_swag = np.zeros((K, nb_models))

    acc_mods = np.zeros((K, nb_models))
    acc_mods_swa = np.zeros((K, nb_models))
    acc_mods_swag = np.zeros((K, nb_models))

    lpd_ens = np.zeros((K, nb_models))
    lpd_swa = np.zeros((K, nb_models))
    lpd_swag = np.zeros((K, nb_models))

    lpd_mods = np.zeros((K, nb_models))
    lpd_mods_swa = np.zeros((K, nb_models))
    lpd_mods_swag = np.zeros((K, nb_models))


    for fold, (train_index, test_index) in enumerate(split_gen):

        metrics = get_acc_fold(X, y, train_index, test_index, batch_size, nb_classes,
                                                       nb_models, fold+1, run_name, swag_params)

        acc_ens[fold, :] = metrics["acc"]
        acc_swa[fold,:] = metrics["acc_swa"]
        acc_swag[fold,:] = metrics["acc_swag"]

        acc_mods[fold,:] = metrics["single_acc"]
        acc_mods_swa[fold,:] = metrics["single_acc_swa"]
        acc_mods_swag[fold,:] = metrics["single_acc_swag"]

        lpd_ens[fold,:] = metrics["log_pred_dens"]
        lpd_swa[fold,:] = metrics["log_pred_dens_swa"]
        lpd_swag[fold,:] = metrics["log_pred_dens_swag"]

        lpd_mods[fold,:] = metrics["single_log_preds"]
        lpd_mods_swa[fold,:] = metrics["single_log_preds_swa"]
        lpd_mods_swag[fold,:] = metrics["single_log_preds_swag"]


    avg_acc_ens = np.mean(acc_ens, axis=0)
    avg_acc_swa = np.mean(acc_swa, axis=0)
    avg_acc_swag = np.mean(acc_swag, axis=0)

    avg_acc_mods = np.mean(acc_mods, axis=0)
    avg_acc_mods_swa = np.mean(acc_mods_swa, axis=0)
    avg_acc_mods_swag = np.mean(acc_mods_swag, axis=0)

    avg_lpd_ens = np.mean(lpd_ens, axis=0)
    avg_lpd_swa = np.mean(lpd_swa, axis=0)
    avg_lpd_swag = np.mean(lpd_swag, axis=0)

    avg_lpd_mods = np.mean(lpd_mods, axis=0)
    avg_lpd_mods_swa = np.mean(lpd_mods_swa, axis=0)
    avg_lpd_mods_swag = np.mean(lpd_mods_swag, axis=0)

    ste_acc_ens = np.std(acc_ens, axis=0)/np.sqrt(K)
    ste_acc_swa = np.std(acc_swa, axis=0)/np.sqrt(K)
    ste_acc_swag = np.std(acc_swag, axis=0)/np.sqrt(K)

    ste_lpd_ens = np.std(lpd_ens, axis=0)/np.sqrt(K)
    ste_lpd_swa = np.std(lpd_swa, axis=0)/np.sqrt(K)
    ste_lpd_swag = np.std(lpd_swag, axis=0)/np.sqrt(K)

    fig, ax = plt.subplots(1,2, figsize=(20,10))

    ax[0].errorbar(list(range(1, nb_models + 1)), avg_acc_ens, yerr=ste_acc_ens, c='C0', label="ensemble", alpha=alpha)
    ax[0].errorbar(list(range(1, nb_models + 1)), avg_acc_swa, yerr=ste_acc_swa, c='C2', label="swa", alpha=alpha)
    ax[0].errorbar(list(range(1, nb_models + 1)), avg_acc_swag, yerr=ste_acc_swag, c='C4', label="swag", alpha=alpha)

    ax[0].plot(list(range(1, nb_models + 1)), avg_acc_mods, c='C9', label='single models', alpha=alpha, marker="o")
    ax[0].plot(list(range(1, nb_models + 1)), avg_acc_mods_swa, c='C8', label='single models swa', alpha=alpha, marker="o")
    ax[0].plot(list(range(1, nb_models + 1)), avg_acc_mods_swag, c='C6', label='single models swag', alpha=alpha, marker="o")

    ax[0].axline((1,np.mean(avg_acc_mods)),(10,np.mean(avg_acc_mods)), c='C9', linestyle='--', label='avg single models', alpha=alpha)
    ax[0].axline((1,np.mean(avg_acc_mods_swa)),(10,np.mean(avg_acc_mods_swa)), c='C8', linestyle='--', label='avg single models swa', alpha=alpha)
    ax[0].axline((1,np.mean(avg_acc_mods_swag)),(10,np.mean(avg_acc_mods_swag)), c='C6', linestyle='--', label='avg single models swag', alpha=alpha)

    ax[0].set(xlabel="No. of models", ylabel="Average accuracy", title='Average accuracy of models on test data across folds with errorbars')
    ax[0].legend()

    ax[1].errorbar(list(range(1, nb_models + 1)), avg_lpd_ens, yerr=ste_lpd_ens, c='C0', label="ensemble", alpha=alpha)
    ax[1].errorbar(list(range(1, nb_models + 1)), avg_lpd_swa, yerr=ste_lpd_swa, c='C2', label="swa", alpha=alpha)
    ax[1].errorbar(list(range(1, nb_models + 1)), avg_lpd_swag, yerr=ste_lpd_swag, c='C4', label="swag", alpha=alpha)

    ax[1].plot(list(range(1, nb_models + 1)), avg_lpd_mods, c='C9', label='single models', alpha=alpha, marker="o")
    ax[1].plot(list(range(1, nb_models + 1)), avg_lpd_mods_swa, c='C8', label='single models swa', alpha=alpha, marker="o")
    ax[1].plot(list(range(1, nb_models + 1)), avg_lpd_mods_swag, c='C6', label='single models swag', alpha=alpha, marker="o")

    ax[1].axline((1,np.mean(avg_lpd_mods)),(10,np.mean(avg_lpd_mods)), c='C9', linestyle='--', label='avg single models', alpha=alpha)
    ax[1].axline((1,np.mean(avg_lpd_mods_swa)),(10,np.mean(avg_lpd_mods_swa)), c='C8', linestyle='--', label='avg single models swa', alpha=alpha)
    ax[1].axline((1,np.mean(avg_lpd_mods_swag)),(10,np.mean(avg_lpd_mods_swag)), c='C6', linestyle='--', label='avg single models swag', alpha=alpha)

    ax[1].set(xlabel="No. of models", ylabel="Average log PD", title='Average log pred dens of models on test data across folds with errorbars')
    ax[1].legend()

    fig.savefig(f"./figures/"+run_name[1:]+f"_avg_acc_fols_w-errorbars.pdf", bbox_inches='tight', format='pdf')

