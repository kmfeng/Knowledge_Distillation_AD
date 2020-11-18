import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision.datasets import ImageFolder
from PIL import Image


def load_data(config):
    normal_class = config['normal_class']
    batch_size = config['batch_size']
    img_size = 0#config['image_size']

    if config['dataset_name'] in ['cifar10']:
        img_transform = transforms.Compose([
            transforms.Resize((256, 256), Image.ANTIALIAS),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))])

        os.makedirs("./Dataset/train/CIFAR10", exist_ok=True)
        dataset = CIFAR10('./Dataset/train/CIFAR10', train=True, download=True, transform=img_transform)
        print("Cifar10 DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/test/CIFAR10", exist_ok=True)
        test_set = CIFAR10("./Dataset/test/CIFAR10", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif config['dataset_name'] in ['mnist']:
        img_transform = transforms.Compose([
            #     transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor()
            #  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        ])

        os.makedirs("./Dataset/train/MNIST", exist_ok=True)
        dataset = MNIST('./Dataset/train/MNIST', train=True, download=True, transform=img_transform)
        print("MNIST DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/test/MNIST", exist_ok=True)
        test_set = MNIST("./Dataset/test/MNIST", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif config['dataset_name'] in ['fashionmnist']:
        img_transform = transforms.Compose([
            #     transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor()
            #  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        ])

        os.makedirs("./train/FashionMNIST", exist_ok=True)
        dataset = FashionMNIST('./train/FashionMNIST', train=True, download=True, transform=img_transform)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]

        train_set, val_set = torch.utils.data.random_split(dataset, [dataset.data.shape[0] - 851, 851])

        os.makedirs("./test/FashionMNIST", exist_ok=True)
        test_set = FashionMNIST("./test/FashionMNIST", train=False, download=True, transform=img_transform)

    elif config['dataset_name'] in ['MVTec']:
        data_path = 'Dataset/MVTec/' + normal_class + '/train'
        data_list = []

        orig_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

        orig_dataset = ImageFolder(root=data_path, transform=orig_transform)

        train_orig, val_set = torch.utils.data.random_split(orig_dataset, [len(orig_dataset) - 25, 25])
        data_list.append(train_orig)

        for i in range(3):
            img_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomAffine(0, scale=(1.05, 1.2)),
                transforms.ToTensor()])

            dataset = ImageFolder(root=data_path, transform=img_transform)
            data_list.append(dataset)

        dataset = ConcatDataset(data_list)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_dataset_array = next(iter(train_loader))[0]
        train_set = TensorDataset(train_dataset_array)

        test_data_path = '/content/' + normal_class + '/test'
        test_set = ImageFolder(root=test_data_path, transform=orig_transform)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
    )

    return train_dataloader, test_dataloader
