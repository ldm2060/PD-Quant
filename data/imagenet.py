import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def build_imagenet_data(data_path: str = '', input_size: int = 224, batch_size: int = 64, workers: int = 4):
    print('==> 使用Pytorch数据集')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 数据归一化
                                     std=[0.229, 0.224, 0.225])
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),  # 随机裁剪并调整大小
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),  # 转换为张量
            normalize,  # 归一化
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,  # 创建训练数据加载器
        num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),  # 调整大小
            transforms.CenterCrop(input_size),  # 中心裁剪
            transforms.ToTensor(),  # 转换为张量
            normalize,  # 归一化
        ])),
        batch_size=batch_size, shuffle=False,  # 创建验证数据加载器
        num_workers=workers, pin_memory=True)
    return train_loader, val_loader
