#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import torch.distributed as dist
import os
from torch.multiprocessing import Process, Array
import pickle

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from dgc.dgc import DGC
import sys
from grace_dl.dist.communicator.allgather import Allgather
from grace_dl.dist.compressor.topk import TopKCompressor
from grace_dl.dist.memory.residual import ResidualMemory

# parse args
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

def partition_dataset():
    """
    :return
    dataset_train: dataset
    dict_users: list of data index for each user. e.g. 100 indexes with 600 data points index
    """
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    return dataset_train, dataset_test, dict_users

def build_model():
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    return net_glob

def init_processing(rank, size, fn, lost_train, acc_train, dataset_train, idxs_users, net_glob, grc, backend='gloo'):
    """initiale each process by indicate where the master node is located(by ip and port) and run main function
    :parameter
    rank : int , rank of current process
    size : int, overall number of processes
    fn : function, function to run at each node
    backend : string, name of the backend for distributed operations
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend=backend, rank=rank, world_size=size)

    fn(rank, size, loss_train, acc_train, dataset_train, idxs_users, net_glob, grc)

def run(rank, world_size, loss_train, acc_train, dataset_train, idxs_users, net_glob, grc):
    # net_glob.load_state_dict(torch.load('net_state_dict.pt'))
    # dgc_trainer = DGC(model=net_glob, rank=rank, size=world_size, momentum=args.momentum, full_update_layers=[4], percentage=args.dgc)
    # dgc_trainer.load_state_dict(torch.load('dgc_state_dict.pt'))
    round = 0
    for i in idxs_users:
        #for each epoch
        idx = dict_users[i[rank]]

        epoch_loss = torch.zeros(1)
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=idx) #create LocalUpdate class
        train_loss = local.train(net=net_glob, world_size=world_size, rank=rank, grc=grc) #train local
        epoch_loss += train_loss
        dist.reduce(epoch_loss, 0, dist.ReduceOp.SUM)

        net_glob.eval()
        train_acc = torch.zeros(1)
        acc, loss = local.inference(net_glob, dataset_train, idx)
        train_acc += acc
        dist.reduce(train_acc, 0, dist.ReduceOp.SUM)

        if rank == 0:
            torch.save(net_glob.state_dict(), 'net_state_dict.pt')
            # torch.save(dgc_trainer.state_dict(), 'dgc_state_dict.pt')
            epoch_loss /= world_size
            train_acc /= world_size
            loss_train[round] = epoch_loss[0]
            acc_train[round] = train_acc[0]
            print('Round {:3d}, Rank {:1d}, Average loss {:.6f}, Average Accuracy {:.2f}%'.format(round, dist.get_rank(), epoch_loss[0], train_acc[0]))
        round+=1
    if rank == 0:
        grc.printr()

import sys
from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size

if __name__ == '__main__':
    dataset_train, dataset_test, dict_users = partition_dataset()
    net_glob = build_model().to('cpu')

    #toggle verbose
    args.verbose=True

    # copy weights
    # torch.save(net_glob.state_dict(), 'net_state_dict.pt')
    # net_glob.load_state_dict(torch.load('net_state_dict.pt'))

    # training
    m = max(int(args.frac * args.num_users), 1)
    loss_train = Array('f', args.epochs)
    acc_train = Array('f', args.epochs)
    # dgc_trainer = DGC(model=net_glob, rank=0, size=m, momentum=args.momentum, full_update_layers=[4], percentage=args.dgc)
    # torch.save(dgc_trainer.state_dict(), 'dgc_state_dict.pt')
    grc = Allgather(TopKCompressor(args.dgc), ResidualMemory(), m)


    # for iter in range(args.epochs):
    #     idxs_users = np.random.choice(range(args.num_users), m, replace=False) #random set of m clients
    idxs_users = [] # size = (epochs * m)
    for _ in range(args.epochs):
        mRand = np.random.choice(range(args.num_users), m, replace=False) #random set of m clients
        idxs_users.append(mRand)

    processes = []

    for i in range(m):
        p = Process(target=init_processing, args=(i, m, run, loss_train, acc_train, dataset_train, idxs_users, net_glob, grc))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # testing
    net_glob.load_state_dict(torch.load('net_state_dict.pt'))
    # torch.save(net_glob, 'net_glob.pt')
    net_glob.eval()
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Avg Train Accuracy: {:.2f}".format(acc_train[-1]))
    print("Testing Accuracy: {:.2f}".format(acc_test))

    #Saving the objects train_loss and train_accuracy:
    file_name = "/save/pickle/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl".\
    format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs)

    #plot loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(loss_train)), loss_train, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}_E{}_B{}_D{}_loss.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs, args.dgc))

    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(acc_train)), acc_train, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}_E{}_B{}_D{}_acc.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs, args.dgc))