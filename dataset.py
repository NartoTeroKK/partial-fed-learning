from typing import List
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import random_split, DataLoader, ConcatDataset

from fedlab.contrib.dataset.femnist import FemnistDataset


def get_mnist(data_path: str='./data'):

    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(data_path, train=True, download=True, transform=transform)
    testset = MNIST(data_path, train=False, download=True, transform=transform)

    return trainset, testset


def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
     
    trainset, testset = get_mnist()

    # Split trainset into 'num_partitions' trainsets
    num_images = len(trainset) // num_partitions
    partition_len = [num_images] * num_partitions
    trainsets = random_split(trainset, partition_len, generator=torch.Generator().manual_seed(2023))

    # Create dataloaders with train+val support
    trainloaders = []
    valloaders = []

    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(num_total * val_ratio)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], generator=torch.Generator().manual_seed(2023))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader


def prepare_comb_dataset(num_partitions: int, 
                    batch_size: int,
                    val_ratio: float = 0.1):
    
    num_partitions *= 2

    trainset, testset = get_mnist()

    # Split trainset into 'num_partitions' trainsets

    num_images = len(trainset) // num_partitions
    partition_len = [num_images] * num_partitions
    trainsets = random_split(trainset, partition_len, generator=torch.Generator().manual_seed(2023))

    # Create dataloaders with train+val support

    trainloaders = []
    valloaders = []

    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(num_total * val_ratio)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], generator=torch.Generator().manual_seed(2023))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    original_sizes = [len(loader.dataset) for loader in trainloaders]
    print("original sizes: ", original_sizes)
    x = num_partitions
    trainloaders[x//2-1] = combine_loaders(trainloaders, batch_size, x)
    valloaders[x//2-1] = combine_loaders(valloaders, batch_size, x, False)

    new_trainloaders = trainloaders[:x//2]
    new_valloaders = valloaders[:x//2]
    new_sizes = [len(loader.dataset) for loader in new_trainloaders]
    print("new sizes: ", new_sizes)

    testloader = DataLoader(testset, batch_size=128)

    return new_trainloaders, new_valloaders, testloader


def combine_loaders(loaders: List[DataLoader], batch_size: int, x: int, shuffle: bool = True):
        datasets = []
        for i in range(x//2-1, x):
            datasets.append(loaders[i].dataset)

        combined_set = ConcatDataset(datasets)
        combined_loader = DataLoader(combined_set, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    
        return combined_loader


def prepare_dataset_centr(batch_size: int, val_ratio: float = 0.1):
    trainset, testset = get_mnist()

    trainloader, valloader = [], []

    num_total = len(trainset)
    num_val = int(num_total * val_ratio)
    num_train = num_total - num_val

    train, val = random_split(trainset, [num_train, num_val], generator=torch.Generator().manual_seed(2023))

    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader, testloader


def get_mnist_loaders(batch_size: int, data_path: str='./data'):

    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(data_path, train=True, download=True, transform=transform)
    testset = MNIST(data_path, train=False, download=True, transform=transform)
     
    trainloader = torch.utils.data.DataLoader(
        trainset,batch_size=batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        testset,batch_size=128, shuffle=False
    )

    return trainloader, testloader

