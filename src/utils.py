import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid

def get_dataset(args):
    """
    Returns train and test dataset and a user group which is a dict where
    the keys are the user index and the values are the corresponding data
    for each of those users.
    """
    data_dir = "/Users/austineyapp/Documents/REP/Year_4/FYP/data/mnist"

    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)

    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

    #sample training data amongst users
    if args.iid:
        #sample IID user data from MNIST
        user_groups = mnist_iid(train_dataset, args.num_users)
    else:
        #sample non-IID user data from MNIST
        user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups

def average_weights(w):
    """
    Returns the average of the weights
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model       : {args.model}')
    print(f'    Optimizer   : {args.optimizer}')
    print(f'    Learning    : {args.lr}')
    print(f'    Epochs      : {args.epochs}')

    print('    Federated Parameters:')
    if args.iid:
        print('     IID')
    else:
        print('     Non-IID')
    print(f'    Fraction of users   : {args.frac}')
    print(f'    Local batch size:   : {args.local_bs}')
    print(f'    Local epochs        : {args.local_ep}\n')
    return