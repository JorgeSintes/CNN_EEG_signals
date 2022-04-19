import torch
import torch.nn as nn

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        N = 160*4
        self.nb_kernels_t_conv = 40
        self.kernel_size_t_conv = (1,30)

        self.nb_kernels_s_conv = 40
        self.kernel_size_s_conv = (64,1)

        self.kernel_size_pool = (1, 15)

        self.linear_in = 40*(N//15)
        self.linear_mid = 80
        self.linear_out = 4

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
        x = x.view(-1, 1, 64, 640)
        # print(x.shape)
        x = self.t_conv(x)
        # print('After t_conv:', x.shape)
        x = self.s_conv(x)
        # print('After s_conv:', x.shape)
        x = self.pool(x)
        # print('After pool:', x.shape)
        x = x.view(-1, 40*(640//15))
        # print('After flatten:', x.shape)
        x = self.fc1(x)
        # print('After fc1:', x.shape)
        x = self.fc2(x)
        # print('After fc2:', x.shape)

        return x


class Network(torch.nn.Module):
    def __init__(self):

        super(Network, self).__init__()
        self.nb_kernels_c1 = 25
        self.kernel_size_c1 = (1, 11)

        self.nb_kernels_c2 = 25
        self.kernel_size_c2 = (2, 1)

        self.kernel_size_m1 = (1, 3)

        self.nb_kernels_c3 = 50
        self.kernel_size_c3 = (1, 11)

        self.kernel_size_m2 = (1, 3)
        self.nb_kernels_c4 = 100
        self.kernel_size_c4 = (1, 11)

        self.kernel_size_m3 = (1, 3)
        self.nb_kernels_c5 = 200
        self.kernel_size_c5 = (1, 11)

        self.kernel_size_m4 = (1, 2)

        self.conv_L2 = nn.Sequential(  # L1 is the input layer in the paper
            # L2
            nn.Conv2d(in_channels=1,
                      out_channels=self.nb_kernels_c1,
                      kernel_size=self.kernel_size_c1,
                      stride=1,
                      padding=0),
            nn.LeakyReLU(),
            # nn.Dropout2d(
            #     p=0.5,
            #     inplace=True),  #I think inplace should be set to True here
        )
        self.conv_L3 = nn.Sequential(
            nn.Conv2d(in_channels=self.nb_kernels_c1,
                      out_channels=self.nb_kernels_c2,
                      kernel_size=self.kernel_size_c2,
                      stride=1,
                      padding=0),
            # nn.BatchNorm2d(num_features=self.nb_kernels_c2),
            nn.LeakyReLU())
        # L4
        self.mp_L4 = nn.MaxPool2d(kernel_size=self.kernel_size_m1,
                                  stride=(1, 3),
                                  padding=0)

        self.conv_L5 = nn.Sequential(
            nn.Conv2d(in_channels=self.nb_kernels_c2,
                      out_channels=self.nb_kernels_c3,
                      kernel_size=self.kernel_size_c3,
                      stride=1,
                      padding=0),
            nn.LeakyReLU(),
            # nn.Dropout2d(p=0.5, inplace=True)
            )

        self.mp_L6 = nn.MaxPool2d(kernel_size=self.kernel_size_m2,
                                  stride=(1, 3),
                                  padding=0)

        self.conv_L7 = nn.Sequential(
            nn.Conv2d(in_channels=self.nb_kernels_c3,
                      out_channels=self.nb_kernels_c4,
                      kernel_size=self.kernel_size_c4,
                      stride=1,
                      padding=0),
            # nn.BatchNorm2d(num_features=self.nb_kernels_c4),
            nn.LeakyReLU(),
            # nn.Dropout2d(p=0.5, inplace=True),
        )

        self.mp_L8 = nn.MaxPool2d(kernel_size=self.kernel_size_m3,
                                  stride=(1, 3),
                                  padding=0)

        self.conv_L9 = nn.Sequential(
            nn.Conv2d(in_channels=self.nb_kernels_c4,
                      out_channels=self.nb_kernels_c5,
                      kernel_size=self.kernel_size_c5,
                      stride=1,
                      padding=0),
            # nn.BatchNorm2d(num_features=self.nb_kernels_c5),
            nn.LeakyReLU())

        self.mp_L10 = nn.MaxPool2d(kernel_size=self.kernel_size_m4,
                                   stride=(1, 2),
                                   padding=0)

        # L11
        self.fc = nn.Linear(in_features=800, out_features=4)

    def forward(self, x):
        x = x.view(-1, 1, 2, 640)
        x = self.conv_L2(x)
        x = self.conv_L3(x)
        x = self.mp_L4(x)
        x = self.conv_L5(x)
        x = self.mp_L6(x)
        x = self.conv_L7(x)
        x = self.mp_L8(x)
        x = self.conv_L9(x)
        x = self.mp_L10(x)
        x = x.view(-1, 800)
        x = self.fc(x)

        return x
        '''
        self.conv_seq = nn.Sequential(# L1 is the input layer in the paper
            # L2
            nn.Conv2d(in_channels=1, out_channels=self.nb_kernels_c1, kernel_size=self.kernel_size_c1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.5, inplace = True), #I think inplace should be set to True here
            # L3
            nn.Conv2d(in_channels=25, out_channels=self.nb_kernels_c2, kernel_size=self.kernel_size_c2, stride=1, padding=0),
            nn.BatchNorm2d(num_features=self.nb_kernels_c2),
            nn.LeakyReLU(),
            # L4
            nn.MaxPool2d(kernel_size=self.kernel_size_m1, stride=(1,3), padding=0),
            # L5
            nn.Conv2d(in_channels=25, out_channels=self.nb_kernels_c3, kernel_size=self.kernel_size_c3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.5, inplace=True),
            # L6
            nn.MaxPool2d(kernel_size=self.kernel_size_m2, stride=(1,3), padding=0),
            # L7
            nn.Conv2d(in_channels=50, out_channels=self.nb_kernels_c4, kernel_size=self.kernel_size_c4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=self.nb_kernels_c4),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.5, inplace=True),
            # L8
            nn.MaxPool2d(kernel_size=self.kernel_size_m3, stride=(1,3), padding=0),
            # L9
            nn.Conv2d(in_channels=100, out_channels=self.nb_kernels_c5, kernel_size=self.kernel_size_c5, stride=1, padding=0),
            nn.BatchNorm2d(num_features=self.nb_kernels_c5),
            nn.LeakyReLU(),
            # L10
            nn.MaxPool2d(kernel_size=self.kernel_size_m4, stride=(1,3), padding=0),
            )
        '''
