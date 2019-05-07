# Adversarial-Training-Masks
Project final for CSCI 5922

# Download MNIST Training Data
To download the MNIST Training data make sure 
you have a updated version of pytorch and torchvision. You will need both. 

Open a new terminal move to this code directory
``` 
$ cd /Users/yashgandhi/Documents/Adversarial-Training-Masks
$ python
```

Inside of the python terminal
```
>>> import torch
>>> from torchvision import datasets
>>> datasets.MNIST(root='./train_data/', train=True, download=True, transform=None)
>>> datasets.MNIST(root='./test_data/', train=False, download=True, transform=None)
```

There should be two folders in the code directory now. One that says train_data and another that says test_data. data.py has more methods to load and transform the data.
