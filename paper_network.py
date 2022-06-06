import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score


class CNN(torch.nn.Module):
    def __init__(self, nb_classes, n):
        super(CNN, self).__init__()

        #N = 160*4
        self.n = n
        self.nb_kernels_t_conv = 40
        self.kernel_size_t_conv = (1,30)

        self.nb_kernels_s_conv = 40
        self.kernel_size_s_conv = (64,1)

        self.kernel_size_pool = (1, 15)

        self.linear_in = 40*(self.n//15)
        self.linear_mid = 80
        self.linear_out = nb_classes

        self.t_conv = nn.Sequential(
                      nn.Conv2d(in_channels=1,
                                out_channels = self.nb_kernels_t_conv,
                                kernel_size = self.kernel_size_t_conv,
                                stride = 1,
                                padding = 'same'),
                      nn.ReLU(),
                      )

        self.s_conv = nn.Sequential(
                      nn.Conv2d(in_channels = self.nb_kernels_t_conv,
                                out_channels = self.nb_kernels_s_conv,
                                kernel_size = self.kernel_size_s_conv,
                                stride = 1,
                                padding = 'valid'),
                      nn.ReLU(),
                      )

        self.pool = nn.AvgPool2d(kernel_size = self.kernel_size_pool,
                                 stride = self.kernel_size_pool,
                                 padding = 0)

        self.fc1 = nn.Sequential(
                      nn.Linear(in_features = self.linear_in,
                                out_features = self.linear_mid),
                      nn.ReLU(),
                      )

        self.fc2 = nn.Linear(in_features = self.linear_mid,
                             out_features = self.linear_out)

    def forward(self, x):
        # print(x.shape)
        x = x.view(-1, 1, 64, self.n)
        # print(x.shape)
        x = self.t_conv(x)
        # print('After t_conv:', x.shape)
        x = self.s_conv(x)
        # print('After s_conv:', x.shape)
        x = self.pool(x)
        # print('After pool:', x.shape)
        x = x.view(-1, 40*(self.n//15))
        # print('After flatten:', x.shape)
        x = self.fc1(x)
        # print('After fc1:', x.shape)
        x = self.fc2(x)
        # print('After fc2:', x.shape)

        return x


class Ensemble():
    def __init__(self, nb_models, nb_classes=4, signal_length=6*160, output_file=None, run_name="", transfer_to_device=False, k=None, w_init_params=(0,1)):
        self.nb_models = nb_models
        self.nb_classes = nb_classes
        self.output_file = output_file
        self.run_name = run_name
        self.k = k

        if transfer_to_device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        print(f"Using {self.device}")
        if output_file:
            output_file.write(f"Using {self.device}\n")

        self.models = [CNN(nb_classes, signal_length) for _ in range(nb_models)]
        self.criterion = None
        self.optimizers = None

        if nb_models > 1:
            for model in self.models:
                for p in model.parameters():
                    torch.nn.init.normal_(p, mean=w_init_params[0], std=w_init_params[1])

        for model in self.models:
            model.to(self.device)


    def train(self, X_train, y_train, num_epochs, lr=1e-5, wd=0, batch_size=None):
        if not self.criterion:
            self.criterion = torch.nn.CrossEntropyLoss()
            self.criterion = self.criterion.to(self.device)
            self.optimizers = [torch.optim.Adam(model.parameters(), lr=lr,
                                           weight_decay=wd) for model in self.models]

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        if batch_size:
            X_train_batches = torch.split(X_train, batch_size, dim=0)
            y_train_batches = torch.split(y_train, batch_size, dim=0)

        for epoch in range(num_epochs):
            for model, optimizer in zip(self.models, self.optimizers):
                model.train()

                if batch_size:
                    for x, y in zip(X_train_batches, y_train_batches):
                        out = model(x)
                        loss = self.criterion(out, y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                else:
                    out = model(X_train)
                    loss = self.criterion(out, y_train)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # Compute and return training accuracy after training
        return self.test(X_train, y_train, batch_size=batch_size)


    def test(self, X_test, y_test, batch_size=None, models_used=None):
        if not models_used:
            models_used = self.nb_models

        if not self.criterion:
            self.criterion = torch.nn.CrossEntropyLoss()
            self.criterion = self.criterion.to(self.device)

        with torch.no_grad():

            outputs = torch.zeros((models_used, X_test.shape[0], self.nb_classes)).to(self.device)
            test_losses = torch.zeros((models_used))
            X_test = X_test.to(self.device)
            y_test = y_test.to(self.device)

            for i, model in enumerate(self.models[:models_used]):
                model.eval()

                if batch_size:
                    X_test_batches = torch.split(X_test, batch_size, dim=0)

                    idx = 0
                    for x in X_test_batches:
                        out = model(x)
                        outputs[i, idx:idx+x.shape[0], :] = torch.nn.functional.softmax(out, dim=1)
                        idx += x.shape[0]

                else:
                    out = model(X_test)
                    outputs[i, :, :] = torch.nn.functional.softmax(out, dim=1)

                test_losses[i] = self.criterion(outputs[i,:,:], y_test)

            outputs = torch.mean(outputs, dim=0)
            preds = torch.max(outputs, 1)[1].to("cpu")
            true = torch.max(y_test, 1)[1].to("cpu")
            test_preds = list(preds.data.numpy())
            test_true = list(true.data.numpy())
            test_losses.to("cpu")

            # Compute accuracies
            test_acc = accuracy_score(test_true, test_preds)
            test_conf = confusion_matrix(test_true, test_preds)

        return test_losses, test_acc, test_conf


    def train_test_on_the_fly(self, X_train, y_train, X_test, y_test, num_epochs, lr=1e-5, wd=0, batch_size=None, verbose=True):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=lr,
                                       weight_decay=wd) for model in self.models]

        train_accuracies, test_accuracies = [], []
        train_losses = torch.zeros((self.nb_models, num_epochs))
        test_losses  = torch.zeros((self.nb_models, num_epochs))

        for epoch in range(num_epochs):
            train_losses[:, epoch], train_acc, train_conf = self.train(X_train, y_train, 1, lr=lr, batch_size=batch_size)
            test_losses[:, epoch], test_acc, test_conf = self.test(X_test, y_test, batch_size=batch_size)

            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            if verbose:
                print(f"Fold {self.k} Epoch {epoch + 1}: Train Accur {train_acc:.4f}, Test Accur {test_acc:.4f}")
                if (epoch + 1) % 5 == 0:
                    print(f'Train conf. matrix, fold {self.k}, epoch {epoch + 1}: \n{train_conf}')
                    print(f'Test conf. matrix, fold {self.k} epoch {epoch + 1}: \n{test_conf}')

                if self.output_file:
                    self.output_file.write(f"Fold {self.k} Epoch {epoch + 1}: Train Accur {train_acc:.4f}, Test Accur {test_acc:.4f}\n")
                    self.output_file.flush()
                    if (epoch + 1) % 5 == 0:
                        self.output_file.write(f'Train conf. matrix, fold {self.k}, epoch {epoch + 1}: \n{train_conf}\n')
                        self.output_file.write(f'Test conf. matrix, fold {self.k} epoch {epoch + 1}: \n{test_conf}\n')
                        self.output_file.flush()

        return train_losses, test_losses, train_accuracies, test_accuracies, train_conf, test_conf


    def save_weights(self, folder_name = ""):
        state_dicts = {}
        for i, model in enumerate(self.models):
            state_dicts[i] = model.to("cpu").state_dict()

        torch.save(state_dicts, f"./models/" + folder_name + "/ensemble_weights_fold_{self.k}"+self.run_name+".tar")


    def load_weights(self):
        state_dicts = torch.load(f"./models/ensemble_weights_fold_{self.k}"+self.run_name+".tar", map_location=self.device)

        for i, model in enumerate(self.models):
            model.load_state_dict(state_dicts[i])

