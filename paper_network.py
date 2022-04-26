import torch
import torch.nn as nn

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

