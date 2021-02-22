#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.distributed.distributed_c10d import get_world_size
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import numpy as np
import random
from sklearn import metrics
from dgc.dgc import DGC
from models.test import test_img


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        batch_size = int(len(dataset)/self.args.num_users)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        batch_loss = 0
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = Variable(images.to(self.args.device)), Variable(labels.to(self.args.device))
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
            batch_loss += loss
            loss.backward()
            return loss.item()
            # for index, (name, parameter) in enumerate(net.named_parameters()):
            #     grad = parameter.grad.data
            #     # grc.acc(grad)
            #     new_tensor = grc.step(grad, name)
            #     grad.copy_(new_tensor)
        #     optimizer.step()
        #     net.zero_grad()
        # return batch_loss/len(self.ldr_train)

    def inference(self, net, dataset_test, idxs):
        dataset = DatasetSplit(dataset_test, idxs)
        self.args.verbose = False
        acc_test, loss_test = test_img(net, dataset, self.args)
        return acc_test, loss_test