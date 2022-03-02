import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from paper_network import Network


def one_hot(array, separated=False):
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    if separated:
        onehot = onehot.reshape(array.shape[0], -1, len(unique))
    return onehot, unique


def select_channels(channel_list, X, electrodes, separated=False):
    if separated:
        return X[:, :, np.isin(electrodes, channel_list), :]
    else:
        return X[:, np.isin(electrodes, channel_list), :]


def cross_validation_1_layer(X, y_pre, K, lr=1e-5, wd=1, batch_size=64, num_epochs=2000, minibatch=True, separated=True, ordered=True, output_file=None):
    if ordered:
        CV = StratifiedKFold(n_splits=K, shuffle=True, random_state=12)
    else:
        CV = KFold(n_splits=K, shuffle=True, random_state=12)

    train_acc = np.zeros((K, num_epochs))
    test_acc = np.zeros((K, num_epochs))
    losses = np.zeros((K, num_epochs))
    train_conf = np.zeros((K, 4, 4))
    test_conf = np.zeros((K, 4, 4))

    y, encoding = one_hot(y_pre, separated=separated)
    y = torch.from_numpy(y)

    # Making the CV dependent on the shape of X which differs depending on 'separated'
    if ordered:
        split_gen = CV.split(np.zeros((X.shape[1], 1)), y_pre[0,:])
    elif separated:
        split_gen = CV.split(np.zeros((X.shape[1], 1)))

    else:
        split_gen = CV.split(X)

    for i, (train_index, test_index) in enumerate(split_gen):

        if separated:
            # Taking train and test slices over subjects 
            X_train, y_train = X[:, train_index, :, :], y[:, train_index]
            X_test, y_test = X[:, test_index, :, :], y[:, test_index]

            # Reshaping arrays to have all observations in first dimension
            X_train = X_train.reshape(-1, X.shape[2], X.shape[3])
            y_train = y_train.reshape(-1, 4)

            X_test = X_test.reshape(-1, X.shape[2], X.shape[3])
            y_test = y_test.reshape(-1, 4)

        else:
            X_train, y_train = X[train_index, :, :], y[train_index, :]
            X_test, y_test = X[test_index, :, :], y[test_index, :]

        model = Network()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        train_acc[i, :], test_acc[i, :], losses[i, :], train_conf[
            i, :, :], test_conf[i, :, :] = train_test_model(
                model,
                criterion,
                optimizer,
                X_train,
                y_train,
                X_test,
                y_test,
                batch_size,
                num_epochs,
                transfer_to_device=True,
                output_file=output_file,
                k=i+1, minibatch=minibatch)

    return train_acc, test_acc, losses, train_conf, test_conf


def train_test_model(model,
                     criterion,
                     optimizer,
                     X_train,
                     y_train,
                     X_test,
                     y_test,
                     batch_size=100,
                     num_epochs=10,
                     transfer_to_device=True,
                     output_file=None,
                     k=None, minibatch=True):

    if transfer_to_device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device}")
        if output_file:
            output_file.write(f"Using {device}\n")

        model = model.to(device)
        criterion = criterion.to(device)
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_test = X_test.to(device)
        y_test = y_test.to(device)

    if minibatch:

        X_train_batches = torch.split(X_train, batch_size, dim=0)
        y_train_batches = torch.split(y_train, batch_size, dim=0)
        X_test_batches = torch.split(X_test, batch_size, dim=0)
        y_test_batches = torch.split(y_test, batch_size, dim=0)

    train_acc, test_acc = [], []
    losses = []
    for epoch in range(num_epochs):

        # Train
        acum_loss = 0
        model.train()

        if minibatch:

            for x, y in zip(X_train_batches, y_train_batches):
                out = model(x)

                # Compute Loss and gradients
                batch_loss = criterion(out, y)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                acum_loss += batch_loss

            losses.append(acum_loss.to("cpu").data.numpy() / batch_size)

        else:
            out = model(X_train)

            # Compute Loss and gradients
            loss = criterion(out, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.to("cpu").data.numpy() / len(y_train))

        # Evaluate
        with torch.no_grad():
            train_preds, train_true, test_preds, test_true = [], [], [], []
            model.eval()

            if minibatch:
                # First on training set
                for x, y in zip(X_train_batches, y_train_batches):
                    out = model(x)
                    preds = torch.max(out, 1)[1].to("cpu")
                    true = torch.max(y, 1)[1].to("cpu")
                    train_preds += list(preds.data.numpy())
                    train_true += list(true.data.numpy())

                # Then on test
                for x, y in zip(X_test_batches, y_test_batches):
                    out = model(x)
                    preds = torch.max(out, 1)[1].to("cpu")
                    true = torch.max(y, 1)[1].to("cpu")
                    test_preds += list(preds.data.numpy())
                    test_true += list(true.data.numpy())

            else:

                out = model(X_train)
                preds = torch.max(out, 1)[1].to("cpu")
                true = torch.max(y_train, 1)[1].to("cpu")
                train_preds = list(preds.data.numpy())
                train_true = list(true.data.numpy())

                out = model(X_test)
                preds = torch.max(out, 1)[1].to("cpu")
                true = torch.max(y_test, 1)[1].to("cpu")
                test_preds = list(preds.data.numpy())
                test_true = list(true.data.numpy())


            # Compute accuracies
            train_acc_cur = accuracy_score(train_true, train_preds)
            test_acc_cur = accuracy_score(test_true, test_preds)

            train_acc.append(train_acc_cur)
            test_acc.append(test_acc_cur)

            print(
                f"Fold {k} Epoch {epoch + 1}: Train Loss {losses[-1]:.4f}, Train Accur {train_acc_cur:.4f}, Test Accur {test_acc_cur:.4f}"
            )

            if output_file:
                output_file.write(
                    f"Fold {k} Epoch {epoch + 1}: Train Loss {losses[-1]:.4f}, Train Accur {train_acc_cur:.4f}, Test Accur {test_acc_cur:.4f}\n"
                )
                output_file.flush()

            if (epoch + 1) % 5 == 0:
                train_conf = confusion_matrix(train_true, train_preds)
                test_conf = confusion_matrix(test_true, test_preds)
                print(f'Train conf. matrix, fold {k}, epoch {epoch + 1}: \n{train_conf}')
                print(f'Test conf. matrix, fold {k} epoch {epoch + 1}: \n{test_conf}')
                if output_file:
                    output_file.write(
                        f'Train conf. matrix, epoch {epoch + 1}: \n{train_conf}\n'
                    )
                    output_file.write(
                        f'Test conf. matrix, epoch {epoch + 1}: \n{test_conf}\n')
                    output_file.flush()

    train_conf = confusion_matrix(train_true, train_preds)
    test_conf = confusion_matrix(test_true, test_preds)

    if k:
        torch.save({f'fold{k}_epochs{num_epochs}_model_state_dict': model.state_dict()}, './models/' + f'fold{k}_epochs{num_epochs}_model_weights.tar')

    return train_acc, test_acc, losses, train_conf, test_conf
