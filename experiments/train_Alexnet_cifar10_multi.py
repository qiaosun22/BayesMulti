# -*- coding: utf-8 -*-
"""
Created on Dec 12 2023

@author: Qiao Sun
"""

from bayes_opt import BayesianOptimization
from models.Alexnet_cifar10 import DemoModel_multi as DemoModel
from train_utils import check_accuracy
from configs.config_Alexnet_cifar10_multi import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from data_loaders.dataloader_cifar10 import *
from train_utils import train_net
from BayesMulti.utils.common_utils import fix_random_seed
import argparse

def opt_function(p1, p2, p3, p4):
    global best_accu
    net = DemoModel(p1=p1, p2=p2, p3=p3, p4=p4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_adam = optim.Adam(net.parameters(), lr = LR_Adam)
    # scheduler_adam = lr_scheduler.CosineAnnealingLR(optimizer_adam, eta_min = LR_Adam_min, T_max = T_max)
    scheduler_adam = lr_scheduler.ReduceLROnPlateau(optimizer_adam, mode = 'min', factor = factor,
                                                    patience = patience, cooldown = cooldown,
                                                    min_lr = LR_Adam_min)

    net = train_net(net, train_loader, test_loader, criterion, optimizer_adam, num_epochs_Adam, scheduler_adam, plot_flag=1)

    # test_accu = test(net)
    accuracy_train = check_accuracy(train_loader, net)

    if accuracy_train > best_accu:
        best_accu = accuracy_train
        torch.save(net.state_dict(), 'cnn_multi.pth')
    return accuracy_train.cpu()


def parse_args():
    parser = argparse.ArgumentParser(description='Optimize Alexnet on CIFAR10 using Bayesian Optimization')
    parser.add_argument('--weight_save_path', type=str, default='./saved_ckpts/Alexnet_ciar10_multi.pth', help='Model weight save path')
    parser.add_argument('--num_epochs_Adam', type=int, default=40, help='Number of epochs for Adam')
    parser.add_argument('--num_iters_Bayes', type=int, default=10, help='Number of iterations for Bayesian Optimization')
    parser.add_argument('--num_init_points', type=int, default=2, help='Number of initial points for Bayesian Optimization')    
    parser.add_argument('--random_seed', type=int, default=666, help='Random seed')
    args = parser.parse_args()
    parser.add_argument('--verbose_Bayes', type=int, default=2, help='Verbose for Bayesian Optimization, verbose = 1 prints only when a maximum is observed, verbose = 0 is silent')
    return args


if __name__ == "__main__":
    args = parse_args()

    # fix random seed
    fix_random_seed(args.random_seed)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device = {device}')

    num_epochs_Adam = args.num_epochs_Adam


    pbounds = {'p1': (0., 1.), 'p2': (0., 1.), 'p3': (0., 1.), 'p4': (0., 1.)}

    best_accu = 0.
    optimizer = BayesianOptimization(
        f=opt_function,
        pbounds=pbounds,
        verbose=args.verbose_Bayes,  
        random_state=1,
    )
    optimizer.probe(
        params={'p1': 0.5, 'p2': 0.5, 'p3': 0.5, 'p4': 0.5},
        lazy=True,
    )

    # optimizer.subscribe(Events.OPTIMIZATION_STEP, logger_bayes)

    optimizer.maximize(
        init_points=args.num_init_points,
        n_iter=args.num_iters_Bayes,
    )    


