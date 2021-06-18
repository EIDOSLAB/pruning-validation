import os

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


def get_data_loaders(data_dir, batch_size, num_workers, pin_memory):
    """
    Build and returns a torch.utils.data.DataLoader for the torchvision.datasets.CIFAR10 DataSet.
    :param data_dir: Location of the DataSet or where it will be downloaded if not existing.
    :param train_batch_size: DataLoader batch size.
    :param valid_size: If greater than 0 defines a validation DataLoader composed of $valid_size * len(train DataLoader)$ elements.
    :param num_workers: Number of DataLoader workers.
    :param pin_memory: If True, uses pinned memory for the DataLoader.
    :param random_seed: Value for generating random numbers.
    :return: (train DataLoader, validation DataLoader, test DataLoader) if $valid_size > 0$, else (train DataLoader, test DataLoader)
    """
    mean = (0.49139968, 0.48215841, 0.44653091)
    std = (0.2023, 0.1994, 0.2010)
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    data_dir = os.path.join(data_dir)
    
    dataset = CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory, persistent_workers=not pin_memory)
    
    return dataloader
