import argparse


def get_arg_parser():
    AVAILABLE_MODELS = ["lenet300_mnist_1", "lenet300_mnist_2",
                        "lenet5_mnist_1",
                        "lenet5_fashion_1",
                        "resnet32_cifar10_1", "resnet32_cifar10_2", "resnet32_cifar10_3",
                        "resnet18_imagenet_1", "resnet18_imagenet_2", "resnet18_imagenet_3", "resnet18_imagenet_4",
                        "resnet101_imagenet_1", "resnet101_imagenet_2", "resnet101_imagenet_3"]
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, choices=AVAILABLE_MODELS, required=True,
                        help="Neural network architecture.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Folder containing the dataset. Default = data.")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size train. Default = 100.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of workers. Default = 8.")
    parser.add_argument("--device", default="cpu", type=str,
                        help="Device index (cpu or 0 or 1 etc.). Default = cpu.")
    
    return parser


def parse_args():
    return get_arg_parser().parse_args()
