

from typing import Dict
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor
from torch.utils.data import DataLoader, Dataset
from config import Config 

def get_cifar10_dataset(image_size: int) -> Dict[str, Dataset]:
    """Downloads and prepares CIFAR10 dataset with transformations."""
    transforms_dict = {
        "train": Compose([
            RandomCrop(image_size, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
        ]),
        "test": ToTensor(),
    }
    dataset = {}
    for split in ["train", "test"]:
        dataset[split] = CIFAR10(
            root=Config.DATA_ROOT,
            train=(split == "train"),
            download=True,
            transform=transforms_dict[split],
        )
    return dataset

def get_dataloader(dataset: Dict[str, Dataset], batch_size: int, num_workers: int) -> Dict[str, DataLoader]:
    """Creates DataLoaders for training and testing."""
    dataloader_dict = {}
    for split in ['train', 'test']:
        dataloader_dict[split] = DataLoader(
            dataset[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
        )
    return dataloader_dict