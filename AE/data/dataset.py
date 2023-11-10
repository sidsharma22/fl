import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
import torchvision.transforms as transforms
from  .lda import Preprocess 
import numpy as np 
from torchvision.datasets import MNIST
import os


class CustomDataset(Dataset):
    def __init__(self, feature_data, target_labels):
        self.feature_data = feature_data
        self.target_labels = target_labels

    def __len__(self):
        return len(self.feature_data)

    def __getitem__(self, idx):
        feature = self.feature_data[idx]
        target = self.target_labels[idx]
        return feature, target


class Mnist_data():
    def __init__(self, NUM_CLIENTS, IID, BATCH_SIZE):
        self.NUM_CLIENTS = NUM_CLIENTS
        self.BATCH_SIZE = BATCH_SIZE
        self.IID = IID
    
    def load_datasets(self):
        preprocess = Preprocess()
        iid = self.IID
        # Download and transform CIFAR-10 (train and test)
        transform = transforms.Compose(
            [transforms.ToTensor()]#, transforms.Normalize((0.5), (0.5))]
        )
        trainset = MNIST("./dataset", train=True, download=True, transform=transform)
        testset = MNIST("./dataset", train=False, download=True, transform=transform)

        # Split training set into 10 partitions to simulate the individual dataset
        #print("Test Set length ",len(testset))
        data_new = torch.zeros(60000,28, 28)
        count = 0
        for image, _ in trainset:
            data_new[count] = image
            count +=1
            
        partition_size = len(trainset) // self.NUM_CLIENTS
        lengths = [partition_size] * self.NUM_CLIENTS
        if iid:
            datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
            trainloaders = []
            valloaders = []
            for ds in datasets:
                len_val = len(ds) // 10  # 10 % validation set
                len_train = len(ds) - len_val
                lengths = [len_train, len_val]
                ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
                trainloaders.append(DataLoader(ds_train, batch_size=self.BATCH_SIZE, shuffle=False))
                valloaders.append(DataLoader(ds_val, batch_size=self.BATCH_SIZE))
        else:
            #print(len(trainset))
            #print(trainset.data.shape)
            # Till here the same in maintained but then it loses its shape when it comes out of the create_lda_paritions
            flwr_trainset = (data_new, np.array(trainset.targets, dtype=np.int32))
            datasets,_ =  preprocess.create_lda_partitions(
                dataset=flwr_trainset,
                dirichlet_dist= None,
                num_partitions= 10,
                concentration=0.5,
                accept_imbalanced= False,
                seed= 12,
            )
        # Split each partition into train/val and create DataLoader
            trainloaders = []
            valloaders = []
            for ds in datasets:
                len_val = len(ds[0]) // 10  # 10 % validation set
                len_train = len(ds[0]) - len_val
                lengths = [len_train, len_val]
                cd = CustomDataset(ds[0].astype(np.float32),ds[1])
                ds_train, ds_val = random_split(cd, lengths, torch.Generator().manual_seed(42))
                trainloaders.append(DataLoader(ds_train, batch_size=self.BATCH_SIZE, shuffle=False))
                valloaders.append(DataLoader(ds_val, batch_size=self.BATCH_SIZE))
        testloader = DataLoader(testset, batch_size=self.BATCH_SIZE)
        return trainloaders, valloaders, testloader



class Stackoverflow():
    '''
    Stackoverflow dataset the clients for now are fixed to 10.
    The fixation is because the data is generated using tensorflow federated. 
    '''
    def __init__(self, NUM_CLIENTS, BATCH_SIZE):
        self.NUM_CLIENTS = NUM_CLIENTS
        self.BATCH_SIZE = BATCH_SIZE
        
    def create_val_loaders(self):
        valloaders = []
        for i,f in enumerate(os.listdir("data/stackoverflow/test_np/")):
            #print(f)
            feature_data = np.load('data/stackoverflow/test_np/'+f)['x'].astype(np.float32)
            target_labels = np.load('data/stackoverflow/test_np/'+f)['y']
            ds_val = CustomDataset(feature_data,target_labels)
            valloaders.append(DataLoader(ds_val, batch_size=self.BATCH_SIZE))
        return valloaders

    def create_train_loaders(self):
        trainloaders = []
        for i,f in enumerate(os.listdir("data/stackoverflow/train_np/")):
            #print(f)
            feature_data = np.load('data/stackoverflow/train_np/'+f)['x'].astype(np.float32)
            target_labels = np.load('data/stackoverflow/train_np/'+f)['y']
            ds_train = CustomDataset(feature_data,target_labels)
            trainloaders.append(DataLoader(ds_train, batch_size=self.BATCH_SIZE, shuffle=True))
        return trainloaders


    def load_datasets(self):
        # Download and transform MNIST (train and test)
        trainloaders = self.create_train_loaders()
        #print('****')
        valloaders = self.create_val_loaders()
        return trainloaders, valloaders, valloaders[9]