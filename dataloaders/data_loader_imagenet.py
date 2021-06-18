import os

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet


def get_data_loaders(data_dir, batch_size, num_workers, pin_memory):
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    data_dir = os.path.join(data_dir)
    
    dataset = ImageNet(root=data_dir, split="val", transform=transform_test)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory, persistent_workers=not pin_memory)
    
    return dataloader
