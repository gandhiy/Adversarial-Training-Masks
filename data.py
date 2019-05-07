from sklearn import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import torch

class classification(Dataset):
    def __init__(self, samples):
        self.X0, self.Y = samples
        self.X = self.X0
    
    def transform(self,eps, method, distribution='uniform', low=-1.01, high=1.0):
        if(method == 'random'):
            self.X = self.X0
        for i in range(0, len(self.X)):
            if(distribution == 'stdnorm'):
                eta = eps*np.random.randn(self.X[0].shape[0])
            elif(distribution == 'uniform'):
                eta = eps*np.random.uniform(low=low, high=high, size=self.X[0].shape[0])
            self.X[i] += eta

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        #transformed input, original input, label
        return (
            self.X[i].astype(np.float32),  
            self.X0[i].astype(np.float32), 
            self.Y[i].astype(torch.LongTensor) 
        )
    
def _make(samples, num_features=20, num_informative=10, num_classes=5):
    data = datasets.make_classification(
            n_samples=samples,
            n_features=num_features,
            n_informative=num_informative,
            n_clusters_per_class=1,
            n_classes=num_classes,
        )
    return data


def _split(data, ratio=.75):
    train = data[0][:int(ratio*len(data[0]))], data[1][:int(ratio*len(data[0]))]
    test = data[0][int(ratio*len(data[0])):], data[1][int(ratio*len(data[0])):]
    return (train,test)

def transform_MNIST(data, eps, s=(28,28)):
    """
     Method to transform MNIST data with a random mask
    """
    data.transform = transforms.Compose([*data.transform.transforms[:2],
                                            transforms.Lambda(lambda x: x.add(eps*torch.rand(s)))])
    return data


def load(data_class, samples=10000, num_classes=5, num_features = 20, ratio=.75):
    '''
     Load data to be used for training and testing here
    '''
    if(data_class == 'classification'):
        data_train, data_test = _split(_make(samples, num_features = num_features, 
        num_classes=num_classes), ratio=ratio)
        data_train = classification(data_train)
        data_test = classification(data_test)

    elif(data_class == 'MNIST'):
        data_train = MNIST('./train_data', transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))]))
        data_test = MNIST('./test_data', transform = transforms.Compose( 
            [transforms.ToTensor(), 
             transforms.Normalize((0.1307,), (0.3081,))]))
    else:
        print('data class must be either classification or MNIST')
        raise ValueError

    return data_train, data_test

