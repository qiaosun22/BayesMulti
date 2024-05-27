import torchvision
import torchvision.datasets as datasets  # Standard datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation

from configs.config_Alexnet_cifar10_erm import num_workers, batch_size



# train_dataset = datasets.CIFAR10(root = "dataset/", train = True, 
#                                     transform = transforms.Compose([transforms.ToTensor()]),
#                                     download = True)
# test_dataset = datasets.CIFAR10(root = "dataset/", train = False,
#                                     transform = transforms.Compose([transforms.ToTensor()]),
#                                     download = True)


# 导入训练集数据
train_loader = DataLoader(
    torchvision.datasets.CIFAR10(root='dataset/', train=True, transform=torchvision.transforms.Compose([
        # torchvision.transforms.Resize(224, 224),      # 重新设置图片大小
        torchvision.transforms.ToTensor(),      # 将图片转化为tensor
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])         # 进行归一化
    ]), download=False), shuffle=True, batch_size=batch_size
)

# 导入测试集数据
test_loader = DataLoader(
    torchvision.datasets.CIFAR10(root='dataset/', train=False, transform=torchvision.transforms.Compose([
        # torchvision.transforms.Resize(224, 224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=False), shuffle=True, batch_size=batch_size
)

# train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = 1, num_workers = num_workers)
# test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = 0, num_workers = num_workers)
