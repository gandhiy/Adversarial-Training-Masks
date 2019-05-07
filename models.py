import torch
import torch.nn as nn

class LeNet(nn.Module):
    '''
     Use LeNet for MNIST data
     28 -> 25 -> 12 -> 9 -> FC
    '''
    
    def __init__(self, in_channels, num_classes):
        super(LeNet, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels, 6, 3, 1),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 28, 3, 1),
            nn.MaxPool2d(2),

        )
        self.net2 = nn.Sequential(
            nn.Linear(5*5*28, 150),
            nn.Linear(150, num_classes),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.net1(x)
        x = x.view(-1, 5*5*28)
        y = self.net2(x)
        return y
        
class LinearNet(nn.Module):
    def __init__(self, num_feats, num_classes):
        super(LinearNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_feats,100),
            nn.ReLU(),
            
            nn.Linear(100,500),
            nn.ReLU(),
            
            nn.Linear(500,500),
            nn.ReLU(),
            
            nn.Linear(500,100),
            nn.ReLU(),
            
            nn.Linear(100,num_classes),
            nn.Softmax(dim=1)
        )
        
    
    def forward(self, x):
        return self.net(x)

class Mask(nn.Module):
    def __init__(self, input_shape):
        super(Mask, self).__init__()
        self.mask_parameters = nn.Parameter(data=torch.rand(input_shape))
        
    def forward(self, x):
        return self.mask_parameters + x
    
class LinearNet_Generator(nn.Module):
    def __init__(self, num_feats):
        super(LinearNet_Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_feats, 100),
            nn.ReLU(),
            
            nn.Linear(100,500),
            nn.ReLU(),
            
            nn.Linear(500,500),
            nn.ReLU(),
            
            nn.Linear(500,100),
            nn.ReLU(),
            
            nn.Linear(100, num_feats),
        )
        
    def forward(self, x):
        return self.net(x) + x

class LeNet_Generator(nn.Module):
    '''
     Use LeNet for MNIST data
     28 -> 25 -> 12 -> 9 -> FC
    '''
    
    def __init__(self, in_channels):
        super(LeNet_Generator, self).__init__()
        self.down = nn.Sequential(
           nn.Conv2d(in_channels, 6, 4, 2, 1),
           nn.ReLU(),
           nn.Conv2d(6, 28, 4, 2, 1),
           nn.ReLU()

        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(28, 6, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 1, 4, 2, 1),
            nn.ReLU()

        )
    def forward(self, x):
        eps = self.up(self.down(x))
        return x + eps