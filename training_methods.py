from models import LeNet, LeNet_Generator, LinearNet, LinearNet_Generator, Mask
from data import *
from torch.utils.data import DataLoader
from updates import *
from loss_functions import *
import torch.nn as nn


def train_baseline(
    data_type, 
    data,
    samples=10000,
    epochs=100, 
    batch_size=128,
    num_classes=5, 
    opt_net_type='SGD',
    opt_net_args={'lr': 0.27},
    opt_mask_type=None,
    opt_mask_args=None,
    eps=None,
    eps_method=None,
    alpha=None,
    beta=None,
    verbose=False,
    ):
    """
    Baseline classification

     Parameters:
     -----------
     data_type: either 'MNIST' or 'classification' \n
     samples: the number of samples to use for training if using classification \n
     epochs: number of training epochs \n
     batch_size: size of batches for BSGD \n
     num_classes: number of classification classes (set to 10 if using MNIST) \n
     opt_net_type: 'SGD', 'RMS', or 'Adam' \n
     opt_net_args: a dictionary of arguments for the chosen optimizer default = {'lr':0.1} \n
    """

    print(f'ARGS for baseline \n -------- \n samples {samples} \n epochs {epochs}\
         \n batch size {batch_size} \n number of classes {num_classes}\
        \n net optimizer type {opt_net_type} \n net optimizer args {opt_net_args}')
    dataset_train, dataset_test = data

    train = DataLoader(dataset_train, batch_size=batch_size)
    test = DataLoader(dataset_test)
    
    if(data_type == 'classification'):
        net = LinearNet(*dataset_train[0][0].shape, num_classes)
    elif('MNIST'):
        net = LeNet(1, 10)
    else:
        print("Need the correct data type")
        return -1
    
    opt = Optimizer(opt_net_type, net.parameters(), **opt_net_args)
    
    crit = nn.CrossEntropyLoss()

    train_acc = []
    test_acc = []

    for e in range(epochs):
        batch_acc = 0
        total = 0
        for i, vals in enumerate(train):
            if(data_type == 'classification'):
                (_, x, y) = vals
            else:
                (x,y) = vals
            yh = net(x)
            total += len(x)
            batch_acc += torch.argmax(yh, dim=1).eq(y).sum().numpy()
            loss = crit(yh, y)
            opt.zero_grad()
            
            loss.backward()
            opt.step()
        
        train_acc.append(batch_acc/total)
        
        acc = []
        for j, vals in enumerate(test):
            if(data_type == 'classification'):
                (_, x, y) = vals
            else:
                (x,y) = vals
            yh = net(x)

            if(torch.argmax(yh, dim=1) == y):
                acc.append(1)
            else:
                acc.append(0)

        print(f'===> Epoch {e + 1} Network Accuracy {np.mean(acc)}')

        test_acc.append(np.mean(acc))
    
    return test_acc, train_acc, net
            
def train_random_mask(
    data_type, 
    data,
    samples=10000, 
    epochs=100, 
    batch_size=128, 
    num_classes=5, 
    opt_net_type='SGD',
    opt_net_args={'lr': 0.27},
    eps_method='fixed',
    eps = 0.05
    ):
    """
     Random Mask Classification

     Parameters:
     -----------
     data_type: either 'MNIST' or 'classification' \n
     samples: the number of samples to use for training if using classification \n
     epochs: number of training epochs \n
     batch_size: size of batches for BSGD \n
     num_classes: number of classification classes (set to 10 if using MNIST) \n
     opt_net_type: 'SGD', 'RMS', or 'Adam' \n
     opt_net_args: a dictionary of arguments for the chosen optimizer default = {'lr':0.1} \n
     eps_method: 'fixed' method adds a random mask to the previously 
                  used mask 'random' method adds a mask to the original data \n
    eps: parameter to multiply for random mask
    """
    print(f'ARGS for random mask \n -------- \n samples {samples} \n epochs {epochs}\
         \n batch size {batch_size}\
         \n number of classes {num_classes} \n net optimizer type {opt_net_type}\
         \n net optimizer args {opt_net_args} \n eps transform method {eps_method}\
         \n eps parameter {eps}')

    #gather data for training and test
    dataset_train, dataset_test = data

    #a hold variable for the data
    dataset_train_og = dataset_train
    if (data_type == 'classification'):
        #network to test mask performance
        net_mask = LinearNet(*dataset_train[0][0].shape, num_classes)
    elif('MNIST'):
        print('Number of samples and number of classes ignored')
        net_mask = LeNet(1, 10)
    else:
        print("data_type must be either 'MNIST' or 'classification' " )
        return -1

    #optimizer for networks
    opt_mask = Optimizer(opt_net_type, net_mask.parameters(), **opt_net_args)

    #loss function for both networks
    crit1 = nn.CrossEntropyLoss()

    #init test data
    test = DataLoader(dataset_test, shuffle=True)
    
    train_acc = []
    test_acc = []
    #train
    for e in range(0, epochs):
        #define a training run for make_classification data
        batch_acc = 0
        total = 0
        #tranform based on eps method
        if(data_type == 'classification'):
            dataset_train.transform(eps, eps_method)
        else:
            if(eps_method == 'random'):
                #reset the data
                dataset_train = dataset_train_og
            #transform the MNIST data
            dataset_train = transform_MNIST(dataset_train, eps)
        #init training data based on transformed data
        train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        for i, vals in enumerate(train):
            if(data_type == 'classification'):
                (_, x, y) = vals
            else:
                (x, y) = vals
            yh = net_mask(x)
            total += len(x)
            batch_acc += torch.argmax(yh, dim=1).eq(y).sum().numpy()
            loss_mask = crit1(yh, y)
            opt_mask.zero_grad()
            loss_mask.backward()
            opt_mask.step()
        train_acc.append(batch_acc/total)
        acc_mask = []

        for j, vals in enumerate(test):
            if(data_type == 'classification'):
                (_, x, y) = vals
            else:
                (x,y) = vals
            
            net_mask.train(mode=False)
            yh_mask = net_mask(x)
            if(torch.argmax(yh_mask, dim=1) == y):
                acc_mask.append(1)
            else:
                acc_mask.append(0)
        print(f'===> Epoch {e + 1} Mask Accuracy {np.mean(acc_mask)}')
        test_acc.append(np.mean(acc_mask))


    return test_acc, train_acc, net_mask

def train_learned_mask(
    data_type,  
    data,
    samples=10000, 
    epochs=100, 
    batch_size=128, 
    num_classes=5, 
    opt_mask_type='SGD',
    opt_net_type='SGD',
    opt_mask_args={'lr': 0.27}, 
    opt_net_args={'lr': 0.27},
    verbose=True
    ):
    """
     Learned Classification

     Parameters:
     -----------
     data_type: either 'MNIST' or 'classification' \n
     samples: the number of samples to use for training if using classification \n
     epochs: number of training epochs \n
     batch_size: size of batches for BSGD \n
     num_classes: number of classification classes (set to 10 if using MNIST) \n
     opt_mask_type: optimization technique for learning mask parameters ('SGD', 'RMS', or 'Adam') \n
     opt_mask_args: a dictionary of arguments for the mask optimizer default = {'lr': 0.1}
     opt_net_type: optimization technique for network parameters ('SGD', 'RMS', or 'Adam') \n
     opt_net_args: a dictionary of arguments for the chosen optimizer default = {'lr':0.1} \n
     verbose: print out mask parameters

    """
    print(f'ARGS for learned mask \n -------- \n data type {data_type} \n samples {samples}\
         \n epochs {epochs} \n batch size {batch_size}\
         \n number of classes {num_classes} \n mask optimizer type {opt_mask_type}\
         \n mask optimizer args {opt_mask_args} \n net optimizer type {opt_net_type}\
         \n net optimizer args {opt_net_args} \n verbose output {verbose}')
    
    dataset_train, dataset_test = data
    dataset_train_og = dataset_train

    if(data_type == 'classification'):
        #learned mask network
        m = Mask(*dataset_train[0][0].shape)
        #networks
        net_mask = LinearNet(*dataset_train[0][0].shape, num_classes)
    elif(data_type == 'MNIST'):
        #mask
        m = Mask((dataset_train[0][0][0].shape[0], dataset_train[0][0][0].shape[1]))
        #networks
        net_mask = LeNet(1, 10)
    else:
        print("data_type must be either 'MNIST' or 'classification' ")
        return -1
    
    if(verbose):
        print(f'Initial Mask Parameters {m.mask_parameters}')


    #mask optimizer
    opt_mask = Optimizer(opt_mask_type, m.parameters(), **opt_mask_args)
    #net optimizers
    opt_net_mask = Optimizer(opt_net_type, net_mask.parameters(), **opt_net_args)


    #loss functions
    crit_mask = nn.CrossEntropyLoss()

    test = DataLoader(dataset_test)

    test_acc = []
    train_acc = []
    params = []
    for e in range(0, epochs):
        train = DataLoader(dataset_train, batch_size=batch_size)
        batch_acc = 0
        total = 0
        for i, vals in enumerate(train):
            if(data_type == 'classification'):
                (_, x, y) = vals
            else:
                (x,y) = vals
            
            if(e != 0):
                x = m(x)

            yh = net_mask(x)
            batch_acc += torch.argmax(yh, dim=1).eq(y).sum().numpy()
            total += len(x)
            loss_net_mask = crit_mask(yh, y)

            opt_net_mask.zero_grad()

            loss_net_mask.backward(retain_graph=True)

            opt_net_mask.step()
        
        train_acc.append(batch_acc/total)
        loss_mask = mask_loss_l2(loss_net_mask, m.mask_parameters)
        opt_mask.zero_grad()
        loss_mask.backward()
        opt_mask.step()
    
        acc_mask = []
        for j, vals in enumerate(test):
            if(data_type == 'classification'):
                (_, x, y) = vals
            else:
                (x, y) = vals

            net_mask.train(mode=False)

            #testing on non transformed data
            y_mask = net_mask(x)

            if(torch.argmax(y_mask, dim=1) == y):
                acc_mask.append(1)
            else:
                acc_mask.append(0)
        
        print(f'===> Epoch {e + 1} Mask Accuracy {np.mean(acc_mask)}')
        test_acc.append(np.mean(acc_mask))
        params.append(m.mask_parameters.detach().numpy())



    if(verbose):
        print(f'Trained Mask Parameters {m.mask_parameters}')
    
    return test_acc, train_acc, (net_mask, params)

def train_mask_generator(
    data_type, 
    data,
    num_features,
    samples=10000, 
    epochs=100, 
    batch_size=128, 
    num_classes=5, 
    opt_net_type='SGD',
    opt_mask_type='SGD',
    opt_net_args={'lr': 0.27},
    opt_mask_args={'lr': 0.27},
    alpha=1,
    beta=1
    ):
    """
     Train a mask generator

     Parameters:
     -----------
     data_type: either 'MNIST' or 'classification' \n
     samples: the number of samples to use for training if using classification \n
     epochs: number of training epochs \n
     batch_size: size of batches for BSGD \n
     num_classes: number of classification classes (set to 10 if using MNIST) \n
     opt_net_type: 'SGD', 'RMS', or 'Adam' \n
     opt_net_args: a dictionary of arguments for the chosen optimizer default = {'lr':0.1} \n
     opt_mask_type & opt_net_args: for mask network

    """
    print(f'ARGS for mask generator \n -------- \n samples {samples} \n epochs {epochs}\
         \n batch size {batch_size}\
         \n number of classes {num_classes} \n net optimizer type {opt_net_type}\
         \n net optimizer args {opt_net_args} \n mask optimizer type {opt_mask_type}\
         \n mask optimizer args {opt_mask_args}')

    train, test = data
    if(data_type == 'classification'):
        discriminator = LinearNet(*num_features, num_classes)
        generator = LinearNet_Generator(*num_features)
    elif(data_type == 'MNIST'):
        discriminator = LeNet(1, 10)
        generator = LeNet_Generator(1)
    else:
        print("Data type must be 'classification' and 'MNIST' ")
        return -1
    

    #generator optimizer
    dis_opt = Optimizer(opt_net_type, discriminator.parameters(), **opt_net_args)
    gen_opt = Optimizer(opt_mask_type, generator.parameters(), **opt_mask_args)

    #loss function for generator
    classification_loss = nn.CrossEntropyLoss()
    reconstruct_loss = nn.MSELoss()

    loss_dis_array = []
    loss_gen_array = []
    for e in range(0, epochs):
        for i, vals in enumerate(train):
            if(data_type == 'classification'):
                (_,x,y) = vals
            else:
                (x,y) = vals
            
            xe = generator(x)
            yh = discriminator(xe)

            loss_dis = classification_loss(yh, y)
            loss_gen = (alpha/loss_dis) + beta*reconstruct_loss(xe, x)
            
            loss_dis_array.append(loss_dis.detach().numpy())
            loss_gen_array.append(loss_gen.detach().numpy())

            dis_opt.zero_grad()
            gen_opt.zero_grad()

            loss_dis.backward(retain_graph=True)
            loss_gen.backward()

            dis_opt.step()
            gen_opt.step()
        
        print(f'===> Epoch {e + 1} AVG Discriminator Loss {np.mean(loss_dis_array)} AVG Generator Loss {np.mean(loss_gen_array)}')
    return generator       

def train_generated_mask(
    data_type,
    data,
    samples=10000, 
    epochs=100, 
    generator_epochs = 50,
    batch_size=128, 
    num_classes=5, 
    opt_net_type='SGD',
    opt_mask_type='SGD',
    opt_net_args={'lr': 0.27},
    opt_mask_args={'lr': 0.27},
    alpha=1,
    beta=1,
    ):
    """
     Generated Classification

     Parameters:
     -----------
     data_type: either 'MNIST' or 'classification' \n
     samples: the number of samples to use for training if using classification \n
     epochs: number of training epochs \n
     generator_epochs: number of epochs to train a generator
     batch_size: size of batches for BSGD \n
     num_classes: number of classification classes (set to 10 if using MNIST) \n
     opt_net_type: 'SGD', 'RMS', or 'Adam' \n
     opt_net_args: a dictionary of arguments for the chosen optimizer default = {'lr':0.1} \n
     opt_mask_type & opt_net_args: for mask network
     alpha & beta: weight parameters for generator loss function
    """
    print(f'ARGS for generated mask \n -------- \n samples {samples} \n epochs {epochs}\
         \n generator epochs {generator_epochs} \n batch size {batch_size}\
         \n number of classes {num_classes} \n net optimizer type {opt_net_type}\
         \n net optimizer args {opt_net_args} \n mask optimizer type {opt_mask_type}\
         \n mask optimizer args {opt_mask_args} \n alpha {alpha} \n beta {beta}')

    dataset_train, dataset_test = data

    if(data_type == 'classification'):
        maskNet = LinearNet(*dataset_train[0][0].shape, num_classes)
    elif(data_type == 'MNIST'):
        maskNet = LeNet(1, 10)

    train = DataLoader(dataset_train, batch_size=batch_size)
    test = DataLoader(dataset_test)

    

    
    data_generator = (train, test)
    generator = train_mask_generator(data_type = data_type,
        data = data_generator, 
        num_features= dataset_train[0][0].shape, 
        samples = samples, 
        epochs = generator_epochs, batch_size=batch_size, 
        num_classes=num_classes, opt_net_type=opt_net_type,
        opt_mask_type=opt_mask_type, opt_net_args = opt_net_args,
        opt_mask_args=opt_mask_args, alpha = alpha, beta=beta
    )
    
    crit = nn.CrossEntropyLoss()
    opt = Optimizer(opt_net_type, maskNet.parameters(), **opt_net_args)

    test_acc = []
    train_acc = []
    for e in range(epochs):
        batch_acc = 0
        total = 0
        for i, vals in enumerate(train):
            if(data_type == 'classification'):
                (_, x, y) = vals
            else:
                (x,y) = vals
            
            xe = generator(x)
            yh = maskNet(xe)

            batch_acc += torch.argmax(yh, dim=1).eq(y).sum().numpy()
            total += len(x)

            loss = crit(yh, y)

            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
    
        train_acc.append(batch_acc/total)
        acc = []
        for j, vals in enumerate(test):
            if(data_type == 'classification'):
                (_, x, y) = vals
            else:
                (x,y) = vals

            yh = maskNet(x)

            if(torch.argmax(yh, dim=1) == y):
                acc.append(1)
            else:
                acc.append(0)
        
        print(f'===> Epoch {e + 1} Accuracy {np.mean(acc)}')
        test_acc.append(np.mean(acc))
    
    return test_acc, train_acc,  (maskNet, generator)





    

    


                