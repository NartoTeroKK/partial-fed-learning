from typing import List
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import random_split, DataLoader, ConcatDataset

from femnist import FEMNIST

import os
import json

def get_femnist(data_path: str='./data'):

    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    trainset = FEMNIST(root=data_path, train=True, download=True, transform=transform)
    testset = FEMNIST(root=data_path, train=False, download=True, transform=transform)
    print(len(trainset), len(testset))

    return trainset, testset



def get_mnist(data_path: str='./data'):

    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(data_path, train=True, download=True, transform=transform)
    testset = MNIST(data_path, train=False, download=True, transform=transform)
    print(len(trainset), len(testset))

    return trainset, testset


def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
     
    trainset, testset = get_femnist()
    #1072113728 128049152

    # Split trainset into 'num_partitions' trainsets
    num_images = len(trainset) // num_partitions
    remainder = len(trainset) % num_partitions
    partition_len = [num_images] * num_partitions
    for i in range(remainder):
        partition_len[i] += 1

    trainsets = random_split(trainset, partition_len, generator=torch.Generator().manual_seed(2023))

    # Create dataloaders with train+val support
    trainloaders = []
    valloaders = []

    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(num_total * val_ratio)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], generator=torch.Generator().manual_seed(2023))
        del trainset_

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    testloader = DataLoader(testset, batch_size=batch_size, num_workers=2)

    del testset, trainsets

    return trainloaders, valloaders, testloader #128049152 1072113728



'''
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

def get_sampled_femnist(data_path: str='./data/data_test_train'):
    
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')

    val_data = []
    train_data = []

    # Function to process user data and append it to the corresponding data list
    def process_user_data(data_list, file_list, data_dir):
        #for filename in file_list:
            #if filename.endswith('.json'):
                file_path = os.path.join(data_dir, file_list[0])
                with open(file_path, 'r') as file:
                    json_data = json.load(file)

                user_data_list = list(json_data['user_data'].values())
                for i in range(len(user_data_list)):
                    if len(data_list) < len(user_data_list):
                        data_list.append(user_data_list[i])
                    else:
                        for key in user_data_list[i]:
                            data_list[i][key].extend(user_data_list[i][key])

    process_user_data(val_data, os.listdir(test_dir), test_dir)
    process_user_data(train_data, os.listdir(train_dir), train_dir)
    
    def create_testdata(data_list):
        test_data = {key: [] for key in data_list[0]}
        for i in range(len(data_list)):
            for key in data_list[i]:
                test_data[key].extend(data_list[i][key])
        return test_data
    
    test_data = create_testdata(val_data)

    # Convert data to PyTorch tensors
    def convert_to_tensors(data_list):
        tensors = []
        for user_data in data_list:
            user_tensor = {}
            for key in user_data:
                user_tensor[key] = torch.tensor(user_data[key])
            tensors.append(user_tensor)
        return tensors

    #train_tensors = convert_to_tensors(train_data)
    #test_tensors = convert_to_tensors(val_data)

    # Create PyTorch datasets   
    def create_datasets(tensors_list, transform):
        datasets = []

        for i in range(len(tensors_list)):
            dataset = MyFEMNIST(root='',
                                data=tensors_list[i]['x'], 
                                targets=tensors_list[i]['y'], 
                                index=i, 
                                transform=transform)
            datasets.append(dataset)  
        return datasets
    
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainsets = create_datasets(train_data, transform)
    valsets = create_datasets(val_data, transform)
    testset = MyFEMNIST(root='',
                        data=test_data['x'], 
                        targets=test_data['y'], 
                        index=0, 
                        transform=transform)
  
    return trainsets, valsets, testset
        

def dataset_femnist(batch_size: int):
    trainsets, valsets, testset = get_sampled_femnist()

    trainloaders = []
    valloaders = []

    for trainset in trainsets:
        trainloaders.append(DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2))
    for valset in valsets:
        valloaders.append(DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2))
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloaders, valloaders, testloader
'''