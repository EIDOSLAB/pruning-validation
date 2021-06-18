# pruning-validation
Test suite for LOBSTER-pruned models.

## Installation

```bash
git clone https://github.com/EIDOSlab/pruning-validation
cd pruning-validation
pip3 install -r requirements.txt
```

## Usage

Using `test_single.py` is possible to evaluate the chosen model on the 
corresponding dataset (e.g. MNIST for LeNet-300).

All the LOBSTER-trained models can be found at:
https://drive.google.com/drive/u/2/folders/1Kv4CMghY3uLMNP81_YktVJhOxgbgX0Zr

### Example
For example, to the LeNet-300 model with lower error (1.65%), on the MNIST dataset present in the folder ./data/MNIST:

```bash
python3 test_single.py \
--model lenet300_mnist_1 \
--data_dir ./data/MNIST \
--batch_size 1000 \
--workers 8 \
--device 0
```

## Arguments
```bash
-h, --help            show this help message and exit
--model {lenet300_mnist_1,lenet300_mnist_2,lenet5_mnist_1,lenet5_fashion_1,resnet32_cifar10_1,resnet32_cifar10_2,resnet32_cifar10_3,resnet18_imagenet_1,resnet18_imagenet_2,resnet18_imagenet_3,resnet18_imagenet_4,resnet101_imagenet_1,resnet101_imagenet_2,resnet101_imagenet_3}
                    Neural network architecture.
--data_dir DATA_DIR   Folder containing the dataset. Default = data.
--batch_size BATCH_SIZE
                    Batch size train. Default = 100.
--workers WORKERS     Number of workers. Default = 8.
--device DEVICE       Device index (cpu or 0 or 1 etc.). Default = cpu.
```
