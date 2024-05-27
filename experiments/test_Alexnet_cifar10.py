import os
from hardware_noise.weight_mapping import weight_mapping as add_noise_to_weights
import numpy as np
import copy
import torch
from train_utils import *
from BayesMulti.utils.common_utils import fix_random_seed
from data_loaders.dataloader_cifar10 import *
from matplotlib import pyplot as plt
import argparse


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
    file2min_ap = {}
    file2max_ap = {}
    for k in file2ap_dict.keys():
        file2avg_ap[k] = np.mean(list(file2ap_dict[k].values()))
        file2min_ap[k] = np.min(list(file2ap_dict[k].values()))
        file2max_ap[k] = np.max(list(file2ap_dict[k].values()))

    file2sigma = np.load(file2sigma_dir, allow_pickle=True).item()
    sigma2avg_ap = {}
    sigma2min_ap = {}
    sigma2max_ap = {}
    for k in file2avg_ap.keys():
        sigma = file2sigma[k]
        if sigma not in sigma2avg_ap:
            sigma2avg_ap[sigma] = file2avg_ap[k]
            sigma2min_ap[sigma] = file2min_ap[k]
            sigma2max_ap[sigma] = file2max_ap[k]

    return sigma2avg_ap, sigma2min_ap, sigma2max_ap


def transorm2usability(data, file2sigma, file2usability):
    sigma2usability = {file2sigma[k]: file2usability[k] for k, v in file2sigma.items()}
    # print(sigma2usability)
    for i, number in enumerate(data[0, :]):
        # print(sigma2usability[i])
        data[0, i] = sigma2usability[number]

    return data

def test_and_process(model, noise_file_dir, file2sigma_dir, file2usability_dir, transform2usability=True):
    file2ap_dict =  test_noise_robustness(model, noise_file_dir=noise_file_dir, file2sigma_dir=file2sigma_dir)
    sigma2avg_ap, sigma2min_ap, sigma2max_ap = post_process(file2ap_dict, file2sigma_dir=file2sigma_dir)

    data = np.array([list(sigma2avg_ap.keys()), list(sigma2max_ap.values()), list(sigma2min_ap.values()), list(sigma2avg_ap.values())]).T
    
    if transform2usability:
        file2sigma = np.load(file2sigma_dir, allow_pickle=True).item()
        file2usability = np.load(file2usability_dir, allow_pickle=True).item()
        data = transorm2usability(data, file2sigma, file2usability)

    data = data[data[:, 0].argsort()].T
    return data

def plot_robustness(data_erm, data_multi, save_path=None):
    plt.scatter(data_erm[0], data_erm[2], label='ERM')
    plt.fill_between(data_erm[0], data_erm[1], data_erm[2], #上限，下限
        facecolor='blue', #填充颜色
        # edgecolor='red', #边界颜色
        alpha=0.3
    ) #透明度
    plt.scatter(data_multi[0], data_multi[2], label='Multi')
    plt.fill_between(data_multi[0], data_multi[1], data_multi[2], #上限，下限
        facecolor='orange', #填充颜色
        # edgecolor='red', #边界颜色
        alpha=0.3
    ) #透明度
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def Argparser():
    parser = argparse.ArgumentParser(description='Noise robustness test')
    # parser.add_argument('--model', type=str, default='cnn', help='Model name')
    parser.add_argument('--weight_path_erm', type=str, default='cnn_erm.pth', help='ERM Model weight path')
    parser.add_argument('--weight_path_multi', type=str, default='cnn_multi.pth', help='Multinomial Model weight path')
    parser.add_argument('--noise_file_dir', type=str, default='./hardware_noise/hardware_data/', help='Noise file directory')
    parser.add_argument('--file2sigma_dir', type=str, default='./hardware_noise/file2sigma.npy', help='File to sigma dictionary path')
    parser.add_argument('--file2usability_dir', type=str, default='./hardware_noise/file2usability.npy', help='File to usability dictionary path')
    parser.add_argument('--N_trials', type=int, default=10, help='Number of trials')
    parser.add_argument('--seed', type=int, default=666, help='Random seed')
    parser.add_argument('--img_save_path', type=str, default='./tools/saved_imgs/noise_robustness.png', help='Image save directory')
    return parser.parse_args()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = Argparser()
    fix_random_seed(args.seed)

    from models.Alexnet_cifar10 import DemoModel
    model_erm = DemoModel().to(device)
    weight_path_erm = args.weight_path_erm
    model_erm.load_state_dict(torch.load(weight_path_erm))

    from models.Alexnet_cifar10 import DemoModel_multi
    model_multi = DemoModel_multi(p1=0.5, p2=0.5, p3=0.5, p4=0.5).to(device)
    weight_path_multi = args.weight_path_multi
    model_multi.load_state_dict(torch.load(weight_path_multi))

    data_erm = test_and_process(model_erm, noise_file_dir=args.noise_file_dir, file2sigma_dir=args.file2sigma_dir, file2usability_dir=args.file2usability_dir)
    data_multi = test_and_process(model_multi, noise_file_dir=args.noise_file_dir, file2sigma_dir=args.file2sigma_dir, file2usability_dir=args.file2usability_dir)

    plot_robustness(data_erm, data_multi, save_path=args.img_save_path)

