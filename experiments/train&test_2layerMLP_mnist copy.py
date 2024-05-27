# -*- coding: utf-8 -*-
"""
Created on Dec 12 2023

@author: Qiao Sun
"""

import copy

import matplotlib.pyplot as plt
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from bayes_opt import BayesianOptimization

from train_utils import *
from models.twolayerMLP_mnist import DemoModel_multi as DemoModel



# Check the accuracy of entire dataset
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            # if img_quant_flag == 1:
            #     x, _ = my.data_quantization_sym(x, half_level = img_half_level)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples * 100


def train_net(model, train_loader, criterion, optimizer, epoch_num, scheduler = None, plot_flag=0):
    # Train Network
    model_max = copy.deepcopy(model.state_dict())
    accuracy_max = 0
    accuracy = 0
    LR = []
    loss_plt = []
    # accuracy_plt = []
    for epoch in range(epoch_num):
        print(f'Epoch = {epoch}')
        loop = tqdm(train_loader, leave=True)
        for (data, targets) in loop:
            # Get data to cuda if possible
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)

            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
            if scheduler != None:
                # 如果使用ReduceLROnPlateau，执行下面一行代码
                scheduler.step(loss)

                # 如果使用CosineAnnealingLR，执行下面一行代码
                # scheduler.step()
                # pass
            accuracy = (sum(scores.argmax(dim = 1) == targets) / len(targets)).item() * 100

            lr_current = optimizer.param_groups[0]['lr']
            loop.set_postfix(
                accuracy = accuracy,
                loss = loss.item(),
                LR = lr_current
            )
            LR.append(lr_current)
            loss_plt.append(loss.item())
            # if lr_current == LR_Adam_min:
            #     min_LR_iter -= 1
            #     if min_LR_iter <= 0:
            #         break
            if accuracy >= accuracy_max:
                accuracy_max = accuracy
                model_max = copy.deepcopy(model)
        if plot_flag == 1:
            # if epoch % 10 == 0:
            plt.plot(LR)
            plt.show()
            plt.plot(loss_plt)
            plt.show()
        
        accuracy_test = check_accuracy(test_loader, model)
        print(f"Accuracy on test set: {accuracy_test:.2f}")

    return model_max




def opt_function(p1,p2):
    global best_accu
    net = DemoModel(p1=p1, p2=p2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_adam = optim.Adam(net.parameters(), lr = LR_Adam)
    # scheduler_adam = lr_scheduler.CosineAnnealingLR(optimizer_adam, eta_min = LR_Adam_min, T_max = T_max)
    scheduler_adam = lr_scheduler.ReduceLROnPlateau(optimizer_adam, mode = 'min', factor = factor, patience = patience, cooldown = cooldown, min_lr = LR_Adam_min)

    net = train_net(net, train_loader, criterion, optimizer_adam, num_epochs_Adam, scheduler_adam)

    # test_accu = test(net)
    accuracy_train = check_accuracy(train_loader, net)

    if accuracy_train > best_accu:
        best_accu = accuracy_train
        torch.save(net.state_dict(), 'cnn_bayes.pth')
    return accuracy_train.cpu()



def test_noise_robustness(model, noise_file_dir, file2sigma_dir=None, N_trials=10):
    hw_data_files = os.listdir(noise_file_dir)
    if file2sigma_dir:
        file2sigma = np.load(file2sigma_dir, allow_pickle=True).item()
    else:
        raise NotImplementedError

    file2ap_dict = {}
    for f_name in sorted(hw_data_files):
        if f_name.endswith('xlsx'):
            file2ap_dict[f_name] = {}
            print(f_name)
            sigma = file2sigma[f_name]
            for n in range(N_trials):
                model_copy = copy.deepcopy(model)

                add_noise_to_weights(model_copy, sigma, noise_file_dir+f_name)
                print('sigma:{}, evaluate-{}'.format(sigma, n))

                accuracy_test = check_accuracy(test_loader, model_copy)

                file2ap_dict[f_name][n] = accuracy_test.detach().cpu().numpy()

    return file2ap_dict


def post_process(file2ap_dict, file2sigma_dir='./hardware_noise/file2sigma.npy'):
    file2avg_ap = {}
    for k in file2ap_dict.keys():
        file2avg_ap[k] = np.mean(list(file2ap_dict[k].values()))

    file2sigma = np.load(file2sigma_dir, allow_pickle=True).item()
    sigma2avg_ap = {}
    for k in file2avg_ap.keys():
        sigma = file2sigma[k]
        if sigma not in sigma2avg_ap:
            sigma2avg_ap[sigma] = file2avg_ap[k]

    return sigma2avg_ap


if __name__ == "__main__":
    # ======================================== #
    # Settings
    # ======================================== #
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device = {device}')


    # # ======================================== #
    # # Hyperparameters
    # # ======================================== #

    num_workers = 0


    # # LR
    LR_Adam = 3e-3
    LR_Adam_min = LR_Adam / 100
    # # LR Scheduler
    factor = 0.5
    patience = 2e3
    cooldown = 0

    batch_size = 1024
    # num_epochs_Adam = 1000


    # load dataset
    train_dataset = datasets.MNIST(root = "dataset/", train = True, 
                                        transform = transforms.Compose([transforms.ToTensor()]),
                                        download = True)
    test_dataset = datasets.MNIST(root = "dataset/", train = False,
                                        transform = transforms.Compose([transforms.ToTensor()]),
                                        download = True)

    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = 0,
                            num_workers = num_workers)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = 0,
                            num_workers = num_workers)

    num_epochs_Adam = 20


    # ----------------Train Bayes----------------
    # Bounded region of parameter space

    pbounds = {'p1': (0., 1.), 'p2': (0., 1.)}

    best_accu = 0.

    optimizer = BayesianOptimization(
        f=opt_function,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.probe(
        params={'p1': 0.5, 'p2': 0.5},
        lazy=True,
    )

    # optimizer.subscribe(Events.OPTIMIZATION_STEP, logger_bayes)

    optimizer.maximize(
        init_points=3,
        n_iter=10,
    )    

    # ----------------Train ERM----------------

    from models.twolayerMLP_mnist import DemoModel

    net = DemoModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_adam = optim.Adam(net.parameters(), lr = LR_Adam)
    # scheduler_adam = lr_scheduler.CosineAnnealingLR(optimizer_adam, eta_min = LR_Adam_min, T_max = T_max)
    scheduler_adam = lr_scheduler.ReduceLROnPlateau(optimizer_adam, mode = 'min', factor = factor,
                                                    patience = patience, cooldown = cooldown,
                                                    min_lr = LR_Adam_min)


    num_epochs_Adam = 40

    net = train_net(net, train_loader, criterion, optimizer_adam, num_epochs_Adam, scheduler_adam)

    torch.save(net.state_dict(), 'cnn_erm.pth')

    # ----------------Test----------------
    num_workers = 0
    batch_size = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_dataset = datasets.MNIST(root = "dataset/", train = True, 
    #                                     transform = transforms.Compose([transforms.ToTensor()]),
    #                                     download = True)
    test_dataset = datasets.MNIST(root = "dataset/", train = False,
                                        transform = transforms.Compose([transforms.ToTensor()]),
                                        download = True)

    # train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = 0, num_workers = num_workers)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = 0, num_workers = num_workers)

    from model import DemoModel
    model_erm = DemoModel().to(device)
    weight_path_erm = 'cnn_erm.pth'
    model_erm.load_state_dict(torch.load(weight_path_erm))
    file2ap_dict_erm =  test_noise_robustness(model_erm, noise_file_dir='./hardware_noise/hardware_data/', file2sigma_dir='./hardware_noise/file2sigma.npy')
    sigma2avg_ap_erm = post_process(file2ap_dict_erm)

    from model import DemoModel_multi
    model_multi = DemoModel_multi(p1=0.5, p2=0.5).to(device)
    weight_path_multi = 'cnn_multi.pth'
    model_multi.load_state_dict(torch.load(weight_path_multi))
    file2ap_dict_multi =  test_noise_robustness(model_multi, noise_file_dir='./hardware_noise/hardware_data/', file2sigma_dir='./hardware_noise/file2sigma.npy')
    sigma2avg_ap_multi = post_process(file2ap_dict_multi)

    # ----------------Plot----------------

    plt.scatter(sigma2avg_ap_erm.keys(), sigma2avg_ap_erm.values(), label='ERM')
    plt.scatter(sigma2avg_ap_multi.keys(), sigma2avg_ap_multi.values(), label='Multi')
    plt.legend()
    plt.show()
