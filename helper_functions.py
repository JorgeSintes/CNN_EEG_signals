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
    

def get_acc_fold(X, y, train_index, test_index, batch_size, nb_classes, nb_models, fold, run_name, swa_params, alpha):
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

    for i in range(nb_models):
        loss, acc, _ = model.test(X_test, y_test, batch_size, models_used=i+1)
        accuracies.append(acc)

        loss, acc, _ = model.test(X_test, y_test, batch_size, models_used=[i])
        single_accuracies.append(acc)

    if swa_params:
        model.do_the_swa(X_train, y_train, swa_params["num_epochs"], swa_params["lr"], swa_params["K"], c=swa_params["c"], batch_size=batch_size)

        for i in range(nb_models):
            loss, acc, _ = model.test(X_test, y_test, batch_size, models_used=i+1)
            accuracies_swa.append(acc)

            loss, acc, _ = model.test(X_test, y_test, batch_size, models_used=[i])
            single_accuracies_swa.append(acc)
            
    return accuracies, single_accuracies, accuracies_swa, single_accuracies_swa


def plot_ensemble(X, y_pre, K, batch_size, nb_models, nb_classes, fold, run_name, swa_params=None, alpha=0.5):

    CV = StratifiedKFold(n_splits=K, shuffle=True, random_state=12)
    split_gen = CV.split(np.zeros((X.shape[1], 1)), y_pre[0,:])

    y, encoding = one_hot(y_pre)
    y = torch.from_numpy(y)
    
    train_index, test_index = next(itertools.islice(split_gen, fold-1, None))
        
    #for _ in range(fold):
    #    train_index, test_index = next(split_gen)

    accuracies, single_accuracies, accuracies_swa, single_accuracies_swa = get_acc_fold(X, y, train_index, test_index,
                                                                                        batch_size, nb_classes, nb_models,
                                                                                        fold, run_name, swa_params,
                                                                                        alpha)

    fig, ax = plt.subplots(1,1, figsize=(20,10))
    ax.plot(list(range(1, nb_models + 1)), accuracies, 'b', label="Test accuracies")

    for single_acc in single_accuracies:
        ax.plot(list(range(1, nb_models + 1)), [single_acc]*nb_models, 'g', alpha=alpha)

    if swa_params:
        ax.plot(list(range(1, nb_models + 1)), accuracies_swa, 'r', label="Test accuracies after SWA")

        for single_acc in single_accuracies_swa:
            ax.plot(list(range(1, nb_models + 1)), [single_acc]*nb_models, 'orange', alpha=alpha)

    ax.set(xlabel="No. of models", ylabel="Accuracy")
    ax.legend()
    fig.suptitle(f"Ensemble - Fold: {fold}, classes: {nb_classes}")
    fig.savefig(f"./figures/"+run_name[1:]+f"_{fold}_fold.pdf")
    
    
def plot_ensemble_all(X, y_pre, K, batch_size, nb_models, nb_classes, run_name, swa_params=None, alpha=0.5):
    
    CV = StratifiedKFold(n_splits=K, shuffle=True, random_state=12)
    split_gen = CV.split(np.zeros((X.shape[1], 1)), y_pre[0,:])

    y, encoding = one_hot(y_pre)
    y = torch.from_numpy(y)
    
    accuracies_ens_all_folds = []
    accuracies_swa_all_folds = []
    
    for fold, (train_index, test_index) in enumerate(split_gen):
        
        accuracies, _, accuracies_swa,_ = get_acc_fold(X, y, train_index, test_index, batch_size, nb_classes,
                                                       nb_models, fold+1, run_name, swa_params, alpha)
        
        accuracies_ens_all_folds.append(accuracies)
        accuracies_swa_all_folds.append(accuracies)
        
    accuracies_ens_all_folds = np.asarray(accuracies_ens_all_folds)
    accuracies_swa_all_folds = np.asarray(accuracies_swa_all_folds)
    
    avg_accs_ens = np.mean(accuracies_ens_all_folds, axis=0)
    avg_accs_swa = np.mean(accuracies_swa_all_folds, axis=0)
    
    ste_accs_ens = np.std(accuracies_ens_all_folds, axis=0)/np.sqrt(5)
    ste_accs_swa = np.std(accuracies_swa_all_folds, axis=0)/np.sqrt(5)
    
    plt.figure(figsize=(20,10))
    
    # plt.plot(list(range(1, nb_models + 1)), avg_accs_ens, c='b', label='ensemble')
    plt.errorbar(list(range(1, nb_models + 1)), avg_accs_ens, yerr=ste_accs_ens, label="ensemble")
    
    # plt.plot(list(range(1, nb_models + 1)), avg_accs_swa, c='g', label='swag')
    plt.errorbar(list(range(1, nb_models + 1)), avg_accs_swa, yerr=ste_accs_swa, label="swag")
    
    plt.xlabel="No. of models"
    plt.ylabel="Average kaccuracy"
    plt.legend()
    plt.title('Average accuracy of models on test data across folds with errorbars')
    plt.savefig(f"./figures/"+run_name[1:]+f"_avg_acc_fols_w-errorbars.pdf", bbox_inches='tight', format='pdf')
    
