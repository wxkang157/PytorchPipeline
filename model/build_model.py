from .resnet import resnet34
from .vgg import vgg
from .yourmodel import YourModel

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def get_model(cfg, num_classes):
    if cfg.MODEL.NAME == 'resnet34':
        model = resnet34(num_classes)
        if cfg.GLOBAL.PRETRAINED_MODEL:
            pretrained_weights = torch.load(cfg.GLOBAL.PRETRAINED_MODEL)  # 这里的预训练模型是基于ImageNet训练得到的
            load_pretrained_dict = {k: v for k, v in pretrained_weights.items()
                                    if model.state_dict()[k].numel() == v.numel()}  # 加载结构一致的权重
            model.load_state_dict(load_pretrained_dict, strict=False)
    elif cfg.MODEL.NAME == 'vgg':
        model = vgg(num_classes)
        if cfg.GLOBAL.PRETRAINED_MODEL:
            pretrained_weights = torch.load(cfg.GLOBAL.PRETRAINED_MODEL)  # 这里的预训练模型是基于ImageNet训练得到的
            load_pretrained_dict = {k: v for k, v in pretrained_weights.items()
                                    if model.state_dict()[k].numel() == v.numel()}  # 加载结构一致的权重
            model.load_state_dict(load_pretrained_dict, strict=False)
    else:
        model = YourModel(cfg.GLOBAL.PRETRAINED_MODEL, num_classes)
    return model


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    return model.module if is_parallel(model) else model


def parallel_model(model, device, rank, local_rank):
    # DDP mode
    ddp_mode = device.type != 'cpu' and rank != -1
    if ddp_mode:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    return model