from torch import nn

class TemporalConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3), stride=(1), padding = 'same'):
        super(TemporalConvolution, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,kernel_size),
                               stride=stride, padding=padding)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(kernel_size,1),
                               stride=stride, padding=padding)
        
    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        return x
    
class DepthWiseConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = (3), stride=(1), padding = 'valid'):
        super(DepthWiseConvolution, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, groups=in_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1),
                               stride=stride, padding='valid')
        
    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        return x


class EEGNet_lowpower(nn.Module):
    def __init__(self, task_mode, downsampling_factor=2, number_of_channels=64):
        super(EEGNet_lowpower, self).__init__()

        assert task_mode in ['rest_unrest', 'left_right', 'upper_lower',
                             'all_tasks'], "Mode must be set to one of the following: 'rest_unrest', 'left_right', 'upper_lower', 'all_tasks'"

        if task_mode == 'rest_unrest':
            self.two_classes = True
        elif task_mode == 'left_right' or task_mode == 'upper_lower':
            self.n_classes = 3
        elif task_mode == 'all_tasks':
            self.n_classes = 5

        self.downsampling_factor = downsampling_factor
        self.temporal_filter_size = int(128/self.downsampling_factor)
        self.pooling_kernel_size = 8
        self.number_of_channels = number_of_channels

        if self.n_classes is not None:
            self.two_classes = False

        self.temp_conv = TemporalConvolution(in_channels=1, out_channels=8,
                                             kernel_size=(self.temporal_filter_size), stride=(1,1), padding='same')
        self.batch_norm_1 = nn.BatchNorm2d(8)

        self.depthwise_conv = DepthWiseConvolution(in_channels=8, out_channels=16, 
                                                   kernel_size=(self.number_of_channels,1), stride=(1))
        self.batch_norm_2 = nn.BatchNorm2d(16)
        self.elu_1 = nn.ELU()
        self.avg_pool_1 = nn.AvgPool2d((1,self.pooling_kernel_size), stride=(
            self.pooling_kernel_size))
        
        self.separable_conv = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(16,1), stride=(1,1), padding='same')
        self.batch_norm_3 = nn.BatchNorm2d(16)
        self.elu_2 = nn.ELU()
        self.avg_pool_2 = nn.AvgPool2d((1,8), stride=(1,8))

        self.flat = nn.Flatten()
        
        if self.two_classes:
            self.fc = nn.Linear(112, 1)
        else:
            self.fc = nn.Linear(112, self.n_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.temp_conv(x)
        x = self.batch_norm_1(x)

        x = self.depthwise_conv(x)
        x = self.batch_norm_2(x)
        x = self.elu_1(x)
        x = self.avg_pool_1(x)

        x = self.separable_conv(x)
        x = self.batch_norm_3(x)
        x = self.elu_2(x)
        x = self.avg_pool_2(x)

        x = self.flat(x)
        x = self.fc(x)
        if self.two_classes == True:
            x = self.sigmoid(x)  # sigmoid for 2 class, softmax for 4
        else:
            x = self.softmax(x)

        return x