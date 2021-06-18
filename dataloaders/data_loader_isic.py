import os
import random

import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import resized_crop, hflip, vflip, to_tensor
from yaml import CLoader as Loader


class SegmentationDataset(Dataset):
    
    def __init__(self, data_dir, transform=None, input_image_transform=None, mask_transform=None, split=None):
        self.data_dir = data_dir
        self.transform = transform
        self.input_image_transform = input_image_transform
        self.mask_transform = mask_transform
        self.normalize = transforms.Normalize((0.6681, 0.5301, 0.5247), (0.1337, 0.1480, 0.1595))
        self.imgs = []
        self.masks = []
        self.split = split
        
        data_root = os.path.dirname(data_dir)
        
        with open(self.data_dir, 'r') as stream:
            try:
                d = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)
        
        for s in split:
            for i in d['split'][s]:
                self.imgs.append(
                    os.path.join(data_root, d['images'][i]['location']))
                self.masks.append(
                    os.path.join(data_root, d['images'][i]['label']))
    
    def __getitem__(self, index):
        image = Image.open(self.imgs[index])
        mask = Image.open(self.masks[index])
        
        if "training" in self.split:
            i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.])
            image = resized_crop(image, i, j, h, w, [224, 224])
            mask = resized_crop(mask, i, j, h, w, [224, 224])
            
            if random.random() > 0.5:
                image = hflip(image)
                mask = hflip(mask)
            
            if random.random() > 0.5:
                image = vflip(image)
                mask = vflip(mask)
        
        else:
            image = transforms.Resize(256)(image)
            image = transforms.CenterCrop(224)(image)
            
            mask = transforms.Resize(256)(mask)
            mask = transforms.CenterCrop(224)(mask)
        
        image = to_tensor(image)
        mask = to_tensor(mask)
        
        image = self.normalize(image)
        
        return image, mask
    
    def __len__(self):
        return len(self.imgs)


def get_data_loaders_segmentation(data_dir, batch_size, num_workers, pin_memory):
    data_dir = os.path.join(data_dir)
    
    dataset = SegmentationDataset(data_dir, split=["test"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory, persistent_workers=not pin_memory)
    
    return dataloader
