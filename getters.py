import os
from pathlib import Path

import torch
from torchvision import models as torchvision_models
import gdown

from architectures import LeNet300, LeNet5, resnet32, UNet


def get_device(args):
    # device = 'cpu' or '0'
    cpu = args.device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif args.device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {args.device} requested'  # check availability
    
    cuda = not cpu and torch.cuda.is_available()
    
    return torch.device('cuda:0' if cuda else 'cpu')


def get_ckp_path(args):
    if "lenet300" in args.model:
        if "_1" in args.model:
            id = "1t5ME4Cz6ftFA301D00sHI4da9jLZ_qSK"
        elif "_2" in args.model:
            id = "1hZmLLGg0Dz0lw-6-GBoIntZOb9XRtmsl"
    if "lenet5" in args.model:
        if "fashion" in args.model:
            if "_1" in args.model:
                id = "1LqyhzXFgSyJAIRa7kCNEsBJji01fKWlJ"
        else:
            if "_1" in args.model:
                id = "1n_cu7hMcjvAZqmBoL828AYLs1spaQHXA"
    if "resnet32" in args.model:
        if "_1" in args.model:
            id = "1fsGpAexXfVP-le1Oomgjq70kPD_k17k2"
        elif "_2" in args.model:
            id = "1Cr7QL1fq-x7SHn8uhRWmGzrAMqV8bbo2"
        elif "_3" in args.model:
            id = "14jIN-mD0IkVc_gF2sFpOgRtkssPDpvRx"
    if "resnet18" in args.model:
        if "_1" in args.model:
            id = "1x9l8V1bfGpgZEkK78LnAEwoicwzkPn4w"
        elif "_2" in args.model:
            id = "1T-bl13oTPS4ij5392tojPTVnr_v-EJiu"
        elif "_3" in args.model:
            id = "12e6ZZit9rPe5JlgPbCMdL6xogyNewCZv"
        elif "_4" in args.model:
            id = "1PLPbsg43mZNwLvt-K6MdfsOXPid3lRP7"
    if "resnet101" in args.model:
        if "_1" in args.model:
            id = "1NzOyA3bgoOnXPpHvh1JbkUj0uGRnHfKd"
        elif "_2" in args.model:
            id = "1xqKmmiYnxpCU9g4HQKMCS70fEZqZRMhJ"
        elif "_3" in args.model:
            id = "1b49P9cDXsEAo0KQT2NlTWcEe737gA3Br"
    if "unet" in args.model:
        pass
        
    return id


def get_model(args, device):
    model = None
    dummy_input = None
    
    if "lenet300" in args.model:
        model = LeNet300()
        dummy_input = torch.rand(1, 1, 28, 28)
    if "lenet5" in args.model:
        model = LeNet5()
        dummy_input = torch.rand(1, 1, 28, 28)
    if "resnet32" in args.model:
        model = resnet32("A")
        dummy_input = torch.rand(1, 3, 32, 32)
    if "resnet18" in args.model:
        model = torchvision_models.resnet18(False)
        dummy_input = torch.rand(1, 3, 224, 224)
    if "resnet101" in args.model:
        model = torchvision_models.resnet101(False)
        dummy_input = torch.rand(1, 3, 224, 224)
    if "unet" in args.model:
        model = UNet(3, 1)
        dummy_input = torch.rand(1, 3, 224, 224)

    url = 'https://drive.google.com/uc?id={}'.format(get_ckp_path(args))
    output = args.model + ".pt"
    
    if not Path(output).is_file():
        gdown.download(url, output, quiet=False)
        
    model.load_state_dict(torch.load(output, map_location="cpu"))
    
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    
    return model, dummy_input
