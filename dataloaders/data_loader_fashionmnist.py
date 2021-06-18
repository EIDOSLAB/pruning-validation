import os

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST


def get_data_loaders(data_dir, batch_size, num_workers, pin_memory):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3205,))
    ])
    
    data_dir = os.path.join(data_dir)
    
    dataset = FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=not pin_memory)
    
    return dataloader
