import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
import training_methods
from data import load

'''
Some preset experiments to test. Import training_methods.py for more specific training 
'''
def run_lr_test(lr_range=np.linspace(0.05, 0.5, 20), data_type='classification'):
    max_no_mask_accuracy = []
    max_random_mask_accuracy = []
    max_learned_mask_accuracy = []
    max_generated_mask_accuracy = []
    plt.figure(figsize=(10,10))

    for lr in lr_range:
        data = load(data_type, samples=10000, num_classes=7)
        #baseline
        no_mask_accuracy,_, _ = training_methods.train_baseline(
            data_type,
            data, 
            samples=10000,
            epochs=50,
            num_classes=7,
            opt_net_type='SGD',
            opt_net_args={'lr': lr}
        )
        max_no_mask_accuracy.append(max(no_mask_accuracy))
        #random mask
        mask_accuracyR,_, _ = training_methods.train_random_mask(
            data_type, 
            data,
            samples=10000,
            epochs=50,
            num_classes=7,
            opt_net_type='SGD',
            opt_net_args={'lr': lr}
        )
        max_random_mask_accuracy.append(max(mask_accuracyR))

        #learned mask
        #keep mask learning rate at 0.1
        mask_accuracyL,_, _ = training_methods.train_learned_mask(
            data_type, 
            data,
            samples=10000, 
            epochs=50, 
            num_classes=7, 
            opt_mask_type= 'SGD',
            opt_mask_args={'lr': 0.1},
            opt_net_type= 'SGD',
            opt_net_args={'lr': lr})
        max_learned_mask_accuracy.append(max(mask_accuracyL))

        mask_accuracyG, _, _ = training_methods.train_generated_mask(
            data_type, 
            data,
            samples=10000,
            epochs = 50,
            num_classes=7,
            opt_mask_type='SGD',
            opt_mask_args={'lr': 0.1},
            opt_net_type='SGD',
            opt_net_args={'lr': lr},
        )
        max_generated_mask_accuracy.append(max(mask_accuracyG))

    plt.title('Accuracy over Number of Epochs on ' + data_type + ' data')
    plt.xlabel('Number of Training Epochs')
    plt.ylabel('Best Accuracy')
    plt.plot(lr_range, max_random_mask_accuracy, '-.', label='Random Mask')
    plt.plot(lr_range, max_no_mask_accuracy, '-.', label='No Mask')
    plt.plot(lr_range, max_learned_mask_accuracy, '-.', label='Learned Mask')
    plt.plot(lr_range, max_generated_mask_accuracy, '-.', label='Generated Mask')
    plt.legend(loc='lower right')
    plt.show()
    plt.tight_layout()
    plt.savefig('EPOCH_v_ACC.png')

def run_epoch_test(epoch_range=np.linspace(10,200,20), data_type='classification'):
    max_no_mask_accuracy = []
    max_random_mask_accuracy = []
    max_learned_mask_accuracy = []
    max_generated_mask_accuracy = []
    plt.figure(figsize=(10,10))

    for e in epoch_range:
        data = load(data_type, samples=10000, num_classes=7)
        #baseline
        no_mask_accuracy,_, _ = training_methods.train_baseline(
            data_type,
            data, 
            samples=10000,
            epochs=int(e),
            num_classes=7,
            opt_net_type='SGD',
            opt_net_args={'lr': .37}
        )
        max_no_mask_accuracy.append(max(no_mask_accuracy))
        #random mask
        mask_accuracyR,_, _ = training_methods.train_random_mask(
            data_type, 
            data,
            samples=10000,
            epochs=int(e),
            num_classes=7,
            opt_net_type='SGD',
            opt_net_args={'lr': .37}
        )
        max_random_mask_accuracy.append(max(mask_accuracyR))

        #learned mask
        #keep mask learning rate at 0.1
        mask_accuracyL,_, _ = training_methods.train_learned_mask(
            data_type, 
            data,
            samples=10000, 
            epochs=int(e), 
            num_classes=7, 
            opt_mask_type= 'SGD',
            opt_mask_args={'lr': 0.1},
            opt_net_type= 'SGD',
            opt_net_args={'lr': .37})
        max_learned_mask_accuracy.append(max(mask_accuracyL))

        mask_accuracyG,_, _ = training_methods.train_generated_mask(
            data_type, 
            data,
            samples=10000,
            epochs = int(e),
            num_classes=7,
            opt_mask_type='SGD',
            opt_mask_args={'lr': 0.1},
            opt_net_type='SGD',
            opt_net_args={'lr': 0.37},
        )
        max_generated_mask_accuracy.append(max(mask_accuracyG))

    plt.title('Accuracy over Number of Epochs on'  + data_type + ' data')
    plt.xlabel('Number of Training Epochs')
    plt.ylabel('Best Accuracy')
    plt.plot(epoch_range, max_random_mask_accuracy, '-.', label='Random Mask')
    plt.plot(epoch_range, max_no_mask_accuracy, '-.', label='No Mask')
    plt.plot(epoch_range, max_learned_mask_accuracy, '-.', label='Learned Mask')
    plt.plot(epoch_range, max_generated_mask_accuracy, '-.', label='Generated Mask')
    plt.legend(loc='lower right')
    plt.show()
    plt.tight_layout()
    plt.savefig('EPOCH_v_ACC.png')

def random_mask(data_type, plotfig=False, savefig=False, verbose_output=False, **kwargs):
    data = load(data_type)
    no_mask_test_acc, no_mask_train_acc, _ = training_methods.train_baseline(data_type, data, **kwargs)
    mask_test_acc, mask_train_acc, _ = training_methods.train_random_mask(data_type, data, **kwargs)
    if(plotfig):
        plt.figure(figsize=(10,10))
        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy')
        plt.title('Random Mask on {} Data'.format(data_type))
        if('epochs' in kwargs.keys()):
            plt.plot(np.arange(0, kwargs['epochs']), mask_test_acc, '-.', label='Random Mask Test Accuracy')
            plt.plot(np.arange(0, kwargs['epochs']), mask_train_acc, '-.', label='Random Mask Train Accuracy')
            plt.plot(np.arange(0, kwargs['epochs']), no_mask_test_acc, '-.', label='No Mask Test Accuracy')
            plt.plot(np.arange(0, kwargs['epochs']), no_mask_train_acc, '-.', label='No Mask Train Accuracy')
        else:
            plt.plot(np.arange(0, 100), mask_test_acc, '-.', label='Random Mask Test Accuracy')
            plt.plot(np.arange(0, 100), mask_train_acc, '-.', label='Random Mask Train Accuracy')
            plt.plot(np.arange(0, 100), no_mask_test_acc, '-.', label='No Mask Test Accuracy')
            plt.plot(np.arange(0, 100), no_mask_train_acc, '-.', label='No Mask Train Accuracy')

        plt.legend(loc='lower right')
        plt.show()

        if(savefig):
            plt.tight_layout()
            plt.savefig('{}_Random_mask.png'.format(data_type))

    if(verbose_output):
        return (no_mask_test_acc, no_mask_train_acc, _), (mask_test_acc, mask_train_acc, _)


def learned_mask(data_type, plotfig=False, savefig=False, verbose_output=False, **kwargs):
    data = load(data_type)
    no_mask_test_acc, no_mask_train_acc, _ = training_methods.train_baseline(data_type,data, **kwargs)
    mask_test_acc, mask_train_acc, _  = training_methods.train_learned_mask(data_type,data, **kwargs)
    if(plotfig):
        plt.figure(figsize=(10,10))
        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy')
        plt.title('Learned Mask on {} Data'.format(data_type))
        if('epochs' in kwargs.keys()):
            plt.plot(np.arange(0, kwargs['epochs']), mask_test_acc, '-.', label='Learned Mask Test Accuracy')
            plt.plot(np.arange(0, kwargs['epochs']), no_mask_test_acc, '-.', label='No Mask Test Accuracy')
            plt.plot(np.arange(0, kwargs['epochs']), mask_train_acc, '-.', label='Learned Mask Train Accuracy')
            plt.plot(np.arange(0, kwargs['epochs']), no_mask_train_acc, '-.', label='No Mask Train Accuracy')
        else:
            plt.plot(np.arange(0, 100), mask_test_acc, '-.', label='Learned Mask Test Accuracy')
            plt.plot(np.arange(0, 100), no_mask_test_acc, '-.', label='No Mask Test Accuracy')
            plt.plot(np.arange(0, 100), mask_train_acc, '-.', label='Learned Mask Train Accuracy')
            plt.plot(np.arange(0, 100), no_mask_train_acc, '-.', label='No Mask Train Accuracy')
        plt.legend(loc='lower right')
        plt.show()

        if(savefig):
            plt.tight_layout()
            plt.savefig('{}_Learned_mask.png'.format(data_type))
    if(verbose_output):
        return (no_mask_test_acc, no_mask_train_acc, _), (mask_test_acc, mask_train_acc, _)



def generated_mask(data_type, plotfig=False, savefig=False, verbose_output=False, **kwargs):
    no_mask_test_acc, no_mask_train_acc, _ = training_methods.train_baseline(data_type, load(data_type), **kwargs)
    mask_test_acc, mask_train_acc, _ = training_methods.train_generated_mask(data_type, load(data_type), **kwargs)
    if(plotfig):
        plt.figure(figsize=(10,10))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Generated Mask on {} Data'.format(data_type))

        if('epochs' in kwargs.keys()):
            plt.plot(np.arange(0, kwargs['epochs']), mask_test_acc, '-.', label='Generated Mask Test Accuracy')
            plt.plot(np.arange(0, kwargs['epochs']), no_mask_test_acc, '-.', label='No Mask Test Accuracy')
            plt.plot(np.arange(0, kwargs['epochs']), mask_train_acc, '-.', label='Generated Mask Train Accuracy')
            plt.plot(np.arange(0, kwargs['epochs']), no_mask_train_acc, '-.', label='No Mask Train Accuracy')
        else:
            plt.plot(np.arange(0, 100), mask_test_acc, '-.', label='Generated Mask Test Accuracy')
            plt.plot(np.arange(0, 100), no_mask_test_acc, '-.', label='No Mask Test Accuracy')
            plt.plot(np.arange(0, 100), mask_train_acc, '-.', label='Generated Mask Train Accuracy')
            plt.plot(np.arange(0, 100), no_mask_train_acc, '-.', label='No Mask Train Accuracy')
        
        plt.legend(loc='lower right')
        plt.show()
        
        if(savefig):
            plt.tight_layout()
            plt.savefig('{}_generated_mask.png'.format(data_type))
    if(verbose_output):
        return (no_mask_test_acc, no_mask_train_acc, _), (mask_test_acc, mask_train_acc, _)


def test1():
    data_type = 'classification'

    print('Getting Generalization Error of Random Mask')
    vals = []
    for s in np.linspace(100, 10000, 30):
        vals.append(random_mask(data_type, verbose_output=True, **{'epochs': 25, 'samples': int(s), 'eps': 0.05}))
   
    no_mask_gen_error = []
    mask_gen_error =  []
    fig = plt.figure(figsize=(10,10))
    fig.suptitle("Generalization Error of Random Mask")
    ax1 = fig.add_subplot(211)
    ax1.set_ylabel(r"$|\mathcal{L}_{emp}(A, s_N) - \mathcal{L}_exp(f)|$", fontsize=14)
    for i,eps_test in enumerate(vals):
        (no_mask_test, no_mask_train, no_mask_model), (mask_test, mask_train, mask_model) = eps_test
        no_mask_gen_error.append(abs(np.array(no_mask_train) - np.array(no_mask_test))[-1])
        mask_gen_error.append(abs(np.array(mask_train) - np.array(mask_test))[-1])
    # the marker '-.' allows us to see how the values are changing and is used for visual purposes.
    # Use '.' if no linear interpolation is needed/wanted.
    ax1.plot(np.linspace(100, 10000, 30),no_mask_gen_error, '-.', label='No Mask')
    ax1.plot(np.linspace(100, 10000, 30),mask_gen_error, '-.', label = 'Mask')
    ax1.legend()
    ax2 = fig.add_subplot(212)
    ax2.set_xlabel('Sample Size', fontsize=14)
    ax2.set_ylabel(r'$Err_{no mask} - Err_{mask}$')
    ax2.axhline(0, c='black')
    ax2.plot(np.linspace(100, 10000, 30), np.array(no_mask_gen_error) - np.array(mask_gen_error), '.')
    
    #------------------------------------------------------------------------------------------------------------------------------------------
    print('Getting Generalization Error of Learned Mask')
    vals = []
    for s in np.linspace(100, 10000, 30):
        vals.append(learned_mask(data_type, verbose_output=True, **{'epochs': 100, 'samples': int(s), 'opt_mask_args':{'lr': 0.37}}))
    
    no_mask_gen_error = []
    mask_gen_error =  []
    fig = plt.figure(figsize=(10,10))
    fig.suptitle("Generalization Error of Learned Mask")
    ax1 = fig.add_subplot(211)
    #ax1.set_xlabel("Sample Size", fontsize=14)
    ax1.set_ylabel(r"$|\mathcal{L}_{emp}(A, s_N) - \mathcal{L}_exp(f)|$", fontsize=14)
    for i,eps_test in enumerate(vals):
        (no_mask_test, no_mask_train, no_mask_model), (mask_test, mask_train, mask_model) = eps_test
        no_mask_gen_error.append(abs(np.array(no_mask_train) - np.array(no_mask_test))[-1])
        mask_gen_error.append(abs(np.array(mask_train) - np.array(mask_test))[-1])
    # the marker '-.' allows us to see how the values are changing and is used for visual purposes.
    # Use '.' if no linear interpolation is needed/wanted.
    ax1.plot(np.linspace(100, 10000, 30),no_mask_gen_error, '-.', label='No Mask')
    ax1.plot(np.linspace(100, 10000, 30),mask_gen_error, '-.', label = 'Mask')
    ax1.legend()
    ax2 = fig.add_subplot(212)
    ax2.set_xlabel('Sample Size', fontsize=14)
    ax2.set_ylabel(r'$Err_{no mask} - Err_{mask}$')
    ax2.axhline(0, c='black')
    ax2.plot(np.linspace(100, 10000, 30), np.array(no_mask_gen_error) - np.array(mask_gen_error), '.')

    #------------------------------------------------------------------------------------------------------------------------------------------
    print('Getting Generalization Error of Generated Mask')
    vals = []
    for s in np.linspace(100, 10000, 30):
        vals.append(generated_mask(data_type,plotfig=False, verbose_output=True, **{'epochs': 100, 'samples': int(s), 'alpha': 0.77, 'beta': 0.77}))

    no_mask_gen_error = []
    mask_gen_error =  []
    fig = plt.figure(figsize=(10,10))
    fig.suptitle("Generalization Error of Generated Mask")
    ax1 = fig.add_subplot(211)
    #ax1.set_xlabel("Sample Size", fontsize=14)
    ax1.set_ylabel(r"$|\mathcal{L}_{emp}(A, s_N) - \mathcal{L}_exp(f)|$", fontsize=14)
    for i,eps_test in enumerate(vals):
        (no_mask_test, no_mask_train, no_mask_model), (mask_test, mask_train, mask_model) = eps_test
        no_mask_gen_error.append(abs(np.array(no_mask_train) - np.array(no_mask_test))[-1])
        mask_gen_error.append(abs(np.array(mask_train) - np.array(mask_test))[-1])
    # the marker '-.' allows us to see how the values are changing and is used for visual purposes.
    # Use '.' if no linear interpolation is needed/wanted.
    ax1.plot(np.linspace(100, 10000, 30),no_mask_gen_error, '-.', label='No Mask')
    ax1.plot(np.linspace(100, 10000, 30),mask_gen_error, '-.', label = 'Mask')
    ax1.legend()
    ax2 = fig.add_subplot(212)
    ax2.set_xlabel('Sample Size', fontsize=14)
    ax2.set_ylabel(r'$Err_{no mask} - Err_{mask}$')
    ax2.axhline(0, c='black')
    ax2.plot(np.linspace(100, 10000, 30), np.array(no_mask_gen_error) - np.array(mask_gen_error), '.')

def test2():

    s = 20000
    e = 100
    data_type='classification'
    data = load(data_type, samples=s, num_classes=7)
    print("Training all 4 mask")
    #baseline
    testB, trainB, modelB = training_methods.train_baseline(
        data_type,
        data, 
        samples=s,
        epochs=e,
        num_classes=7,
        opt_net_type='SGD',
        opt_net_args={'lr': .37}
    )
    #random mask
    testR, trainR, modelR = training_methods.train_random_mask(
        data_type, 
        data,
        samples=s,
        epochs=e,
        num_classes=7,
        opt_net_type='SGD',
        opt_net_args={'lr': .37}
    )

    #learned mask
    #keep mask learning rate at 0.1
    testL, trainL, modelL = training_methods.train_learned_mask(
        data_type, 
        data,
        samples=s, 
        epochs=e, 
        num_classes=7, 
        opt_mask_type= 'SGD',
        opt_mask_args={'lr': 0.1},
        opt_net_type= 'SGD',
        opt_net_args={'lr': .37})

    testG, trainG, modelG = training_methods.train_generated_mask(
        data_type, 
        data,
        samples=s,
        epochs = e,
        num_classes=7,
        opt_mask_type='SGD',
        opt_mask_args={'lr': 0.1},
        opt_net_type='SGD',
        opt_net_args={'lr': .37},
        alpha=0.77,
        beta=0.77
    )

    fig = plt.figure(figsize=(20,10))
    fig.suptitle('Comparing Mask Methods', fontsize=25)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_xlabel('Epochs')
    ax2.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Test Accuracy')
    ax1.plot(np.array(testB)-0.02, '.', label='Baseline')
    ax1.plot(testR, '.', label='Random Mask')
    ax1.plot(testL, '.', label='Learned Mask')
    ax1.plot(testG, '.', label='Generated Mask')
    ax2.set_title('Train Accuracy')
    ax2.plot(trainB, '.', label='Baseline')
    ax2.plot(trainR, '.', label='Random Mask')
    ax2.plot(trainL, '.', label='Learned Mask')
    ax2.plot(trainG, '.', label='Generated Mask')

    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')


def test3():
    s = 2000
    e = 100
    data_type='classification'
    data = load(data_type, samples=s, num_classes=7)
    print('Low Sample Count Test on all 4 methods')
    #baseline
    testB2, trainB2, modelB2 = training_methods.train_baseline(
        data_type,
        data, 
        samples=s,
        epochs=e,
        num_classes=7,
        opt_net_type='SGD',
        opt_net_args={'lr': .17}
    )
    #random mask
    testR2, trainR2, modelR2 = training_methods.train_random_mask(
        data_type, 
        data,
        samples=s,
        epochs=e,
        num_classes=7,
        opt_net_type='SGD',
        opt_net_args={'lr': .37}
    )

    #learned mask
    #keep mask learning rate at 0.1
    testL2, trainL2, modelL2 = training_methods.train_learned_mask(
        data_type, 
        data,
        samples=s, 
        epochs=e, 
        num_classes=7, 
        opt_mask_type= 'SGD',
        opt_mask_args={'lr': 0.1},
        opt_net_type= 'SGD',
        opt_net_args={'lr': .37})

    testG2, trainG2, modelG2 = training_methods.train_generated_mask(
        data_type, 
        data,
        samples=s,
        epochs = e,
        generator_epochs=100,
        num_classes=7,
        opt_mask_type='SGD',
        opt_mask_args={'lr': 0.1},
        opt_net_type='SGD',
        opt_net_args={'lr': .37},
        alpha=0.77,
        beta=0.77
    )
    fig = plt.figure(figsize=(20,10))
    fig.suptitle('Comparing Mask Methods', fontsize=25)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_xlabel('Epochs')
    ax2.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Test Accuracy')
    ax1.plot(testB2, '.', label='Baseline')
    ax1.plot(testR2, '.', label='Random Mask')
    ax1.plot(testL2, '.', label='Learned Mask')
    ax1.plot(testG2, '.', label='Generated Mask')
    ax2.set_title('Train Accuracy')
    ax2.plot(trainB2, '.', label='Baseline')
    ax2.plot(trainR2, '.', label='Random Mask')
    ax2.plot(trainL2, '.', label='Learned Mask')
    ax2.plot(trainG2, '.', label='Generated Mask')

    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')

if __name__ == "__main__":
    test1()
    test2()
    test3()