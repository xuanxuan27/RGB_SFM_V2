import os
import numpy as np
import matplotlib.pyplot as plt


from torch.utils.data import Dataset
from mlxtend.data import loadlocal_mnist
from typing import Any, Callable, Optional, Tuple
from PIL import Image
import torch

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR10(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        augmentation: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        self.train = train
        self.augmentation = augmentation
        if self.train == True:
            self.data, self.targets = self._load_train_data()
        else:
            self.data, self.targets = self._load_test_data()
        
    def _load_train_data(self):
        images = np.array([])
        labels = []
        for i in range(1,6):
            dict = unpickle(f'./data/cifar-10-batches-py/data_batch_{i}')
            if images.any() == False:
                images = dict[b'data']
            else:
                images = np.concatenate((images, dict[b'data']), axis=0)
            
            if len(labels) == 0:
                labels = dict[b'labels']
            else:
                labels = np.concatenate((labels, dict[b'labels']), axis=0)
        
        images = images.reshape(images.shape[0], 3, 32, 32)
        images = images.transpose(0,2,3,1)
        # labels = np.eye(10)[labels]
        labels = np.array(labels, dtype=np.int64)
        return images, labels
    
    def _load_test_data(self):
        dict = unpickle(f'./data/cifar-10-batches-py/test_batch')
        images = dict[b'data']
        labels = dict[b'labels']
        images = images.reshape(images.shape[0], 3, 32, 32)
        images = images.transpose(0,2,3,1)
        # labels = np.eye(10)[labels]
        labels = np.array(labels, dtype=np.int64)
        return images, labels
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)   

        target = torch.tensor(target, dtype=torch.long) 
        return img, target

    def __len__(self) -> int:
        return len(self.data)
    
    
