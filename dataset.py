import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import random_split, DataLoader

def get_mnist(data_path: str='./data'):

    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(data_path, train=True, download=True, transform=transform)
    testset = MNIST(data_path, train=False, download=True, transform=transform)

    return trainset, testset


def prepare_dataset(num_partitions: int, 
                    batch_size: int,
                    val_ratio: float = 0.1):

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


