import csv
import os

from torch import nn

from args import parse_args
from dataloaders import get_dataloader
from evaluation import test_model, architecture_stat
from getters import get_device, get_model
from losses import SoftJaccardBCEWithLogitsLoss

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    
    for model in ["lenet300_mnist_1", "lenet300_mnist_2",
                  "lenet5_mnist_1",
                  "lenet5_fashion_1",
                  "resnet32_cifar10_1", "resnet32_cifar10_2", "resnet32_cifar10_3",
                  "resnet18_imagenet_1", "resnet18_imagenet_2", "resnet18_imagenet_3", "resnet18_imagenet_4",
                  "resnet101_imagenet_1", "resnet101_imagenet_2", "resnet101_imagenet_3"]:
        args.model = model
        
        if "mnist" in model:
            args.data_dir = "/home/tarta/data/MNIST"
        if "fashion" in model:
            args.data_dir = "/home/tarta/data/FashionMNIST"
        if "cifar10" in model:
            args.data_dir = "/home/tarta/data/CIFAR10"
        if "imagenet" in model:
            args.data_dir = "/home/tarta/data/ImageNet"
        
        # Get device, model and dataset
        device = get_device(args)
        model, dummy_input = get_model(args, device)
        dataloader = get_dataloader(args)
        
        # Define task and loss function
        task = "segmentation" if args.model == "unet" else "classification"
        loss_function = SoftJaccardBCEWithLogitsLoss(8) if task == "segmentation" else nn.CrossEntropyLoss()
        
        # Get model performance and stats
        performance = test_model(model, loss_function, dataloader, device, task, desc="Evaluating model performance")
        stats = architecture_stat(model)
        print(args.model)
        print("Top-1 Error: {:.2f}".format(100 - performance[0]))
        print("Pruned Parameters (%) {:.2f}".format(100 - stats["network_param_non_zero_perc"]))
        
        csv_file_name = "results.csv"
        
        vals = [args.model, 100 - performance[0].item(), 100 - stats["network_param_non_zero_perc"]]
        
        if not os.path.exists(csv_file_name):
            titles = ["model", "top-1 error", "pruned params %"]
            with open(csv_file_name, mode='a') as runs_file:
                writer = csv.writer(runs_file, delimiter=';', lineterminator='\n')
                writer.writerow(titles)
                writer.writerow(vals)
        else:
            with open(csv_file_name, mode='a') as runs_file:
                writer = csv.writer(runs_file, delimiter=';', lineterminator='\n')
                writer.writerow(vals)
