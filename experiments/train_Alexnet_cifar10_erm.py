# -*- coding: utf-8 -*-
"""
Created on Dec 12 2023

@author: Qiao Sun
"""


from models.Alexnet_cifar10 import DemoModel
from Multinomial_clean.configs.config_Alexnet_cifar10_erm import *
from data_loaders.dataloader_cifar10 import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from train_utils import train_net
from BayesMulti.utils.common_utils import fix_random_seed

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train Alexnet on CIFAR10')
    parser.add_argument('--weight_save_path', type=str, default='./saved_ckpts/Alexnet_ciar10_erm.pth', help='Model weight save path')
    parser.add_argument('--num_epochs_Adam', type=int, default=40, help='Number of epochs for Adam')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    fix_random_seed(666)
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = DemoModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_adam = optim.Adam(net.parameters(), lr = LR_Adam)
    # scheduler_adam = lr_scheduler.CosineAnnealingLR(optimizer_adam, eta_min = LR_Adam_min, T_max = T_max)
    scheduler_adam = lr_scheduler.ReduceLROnPlateau(optimizer_adam, mode = 'min', factor = factor,
                                                    patience = patience, cooldown = cooldown,
                                                    min_lr = LR_Adam_min)

    net = train_net(net, train_loader, test_loader, criterion, optimizer_adam, args.num_epochs_Adam, scheduler_adam)

    torch.save(net.state_dict(), args.weight_save_path)
