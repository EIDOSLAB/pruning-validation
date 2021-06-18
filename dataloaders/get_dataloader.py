from .data_loader_cifar10 import get_data_loaders as data_loader_cifar10
from .data_loader_fashionmnist import get_data_loaders as data_loader_fashion_mnist
from .data_loader_imagenet import get_data_loaders as data_loader_imagenet
from .data_loader_isic import get_data_loaders_segmentation as data_loaders_isic_segmentation
from .data_loader_mnist import get_data_loaders as data_loader_mnist


def get_dataloader(args):
    if "mnist" in args.model:
        return data_loader_mnist(args.data_dir, args.batch_size, args.workers, True)
    elif "fashion" in args.model:
        return data_loader_fashion_mnist(args.data_dir, args.batch_size, args.workers, True)
    elif "cifar10" in args.model:
        return data_loader_cifar10(args.data_dir, args.batch_size, args.workers, True)
    elif "imagenet" in args.model:
        return data_loader_imagenet(args.data_dir, args.batch_size, args.workers, True)
    elif "isic" in args.model:
        return data_loaders_isic_segmentation(args.data_dir, args.batch_size, args.workers, True)
    else:
        raise ValueError("Unsupported dataset")
