# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import os
import sys
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,distributed
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from common.util import yaml_parser, init_setting, get_parameter_number, view_dataset, setup_seed, \
                build_optimizer, build_scheduler
from data.myClassDataset import MyDataset
from model.build_model import get_model, parallel_model,de_parallel
from loss.build_loss import get_loss
from val import evaluate


class DummyLog(object):
    def __init__(self):
        pass

    @staticmethod
    def info(msg):
        print(msg)

    @staticmethod
    def error(msg):
        print(msg)

def get_envs():
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    rank = int(os.getenv('RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    return local_rank, rank, world_size


def get_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--yaml', default='config/train.yaml', type=str, help='output model name')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter')

    return parser.parse_args()


def train(cfg):
    logger = DummyLog() # 日志打印
    experiment_dir, checkpoints_dir, tensorboard_dir = init_setting(cfg)    #训练实验的根目录、保存模型的目录、保存日志的目录
    setup_seed(2022)

    # setup enviroment
    if cfg.GLOBAL.DEVICE == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif cfg.GLOBAL.DEVICE:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GLOBAL.DEVICE
        assert torch.cuda.is_available()
    cuda = cfg.GLOBAL.DEVICE != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    rank, local_rank, world_size = get_envs()
    if local_rank != -1: # DDP distriuted mode
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo", \
                init_method='env://', rank=local_rank, world_size=world_size)

    tb_writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(
        'Start Tensorboard with "tensorboard --logdir={}", view at http://localhost:6006/'.format(tensorboard_dir))

    if cfg.GLOBAL.VISUAL_AUGMENTATION:
        view_dataset(experiment_dir, cfg)   # 数据增强可视化
    
    train_dataset = MyDataset(cfg=cfg, mode='train')
    train_sampler = (None if rank == -1 else distributed.DistributedSampler(train_dataset, shuffle=True))
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCHSIZE_PER_CARD // world_size,
                              shuffle=True and train_sampler is None,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              sampler=train_sampler,
                              drop_last=cfg.TRAIN.DROP_LAST,
                              collate_fn=train_dataset.collate_fn)
    
    num_classes = len(os.listdir(cfg.GLOBAL.TRAIN_DIR))   # 动物五分类数据集
    logger.info('number of dataset classes : {}'.format(num_classes))   # 这里默认train val的类别数目一致，就没有分开单独打印
    
    if rank in [-1, 0]:
        val_dataset = MyDataset(cfg=cfg, mode='val')
        val_loader = DataLoader(val_dataset,
                                batch_size=cfg.TRAIN.BATCHSIZE_PER_CARD,
                                shuffle=False,
                                num_workers=cfg.TRAIN.NUM_WORKERS,
                                drop_last=cfg.TRAIN.DROP_LAST,
                                collate_fn=val_dataset.collate_fn)
    
    net = get_model(cfg, num_classes)   # 创建模型
    logger.info(get_parameter_number(net))  # 计算模型参数量
    net = net.to(device)
    
    optimizer = build_optimizer(net, cfg, logger)   # 创建优化器
    scheduler = build_scheduler(optimizer, cfg)   # 创建学习率
    loss_function = get_loss(cfg)   # 创建损失函数
    start_epoch = 0
    
    if cfg.GLOBAL.RESUME:
        checkpoint = torch.load(cfg.GLOBAL.RESUME_PATH)  # 这里的checkpoint是项目中训练好的模型
        logger.info('loading checkpoint from {}'.format(cfg.GLOBAL.RESUME_PATH)) # 读取状态字典中的信息
        start_epoch = checkpoint['epoch']
        state_dict = checkpoint['state_dict_backbone']
        net.load_state_dict(state_dict, strict=False)
        state_optimizer = checkpoint['state_optimizer']
        optimizer.load_state_dict(state_optimizer)
        state_lr_scheduler = checkpoint['state_lr_scheduler']
        scheduler.load_state_dict(state_lr_scheduler)


    # create parallel model
    net = parallel_model(net, device, rank, local_rank)
    
    scaler = GradScaler(enabled=cfg.GLOBAL.USE_AMP)   # 用于混合精度训练
    
    pre_acc = 0.0
    early_stop_patience = 0
    for epoch in range(start_epoch, cfg.GLOBAL.EPOCH_NUM):
        net.train()
        if rank != -1:
            train_loader.sampler.set_epoch(epoch)
        mean_loss = torch.zeros(1).to(device)
        data_loader = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(data_loader):
            optimizer.zero_grad()
            with autocast(enabled=cfg.GLOBAL.USE_AMP):    # 用于混合精度训练
                images, labels = data
                pred = net(images.to(device))
                loss = loss_function(pred, labels.to(device))
                if rank != -1:
                    loss *= world_size
                mean_loss = (mean_loss * step + loss.detach()) / (step + 1)
                data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))
                if not torch.isfinite(loss):
                    logger.info('WARNING: non-finite loss, ending training ', loss)
                    sys.exit(1)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        scheduler.step()

        if epoch % cfg.GLOBAL.VAL_EPOCH_STEP == 0 and rank in [-1, 0]:
            acc = evaluate(model=net, data_loader=val_loader, device=device)    # 评估验证集的acc
            logger.info("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], acc, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        if epoch % cfg.GLOBAL.SAVE_EPOCH_STEP == 0 and rank in [-1, 0]:
            checkpoint = {
                "epoch": epoch,
                "state_dict_backbone": deepcopy(de_parallel(net)).state_dict(),
                "state_optimizer": optimizer.state_dict(),
                "state_lr_scheduler": scheduler.state_dict()
            }   # 保存状态字典
            torch.save(checkpoint, checkpoints_dir / "model-{}.pth".format(epoch))
            if acc > pre_acc:
                torch.save(checkpoint, checkpoints_dir / "best.pth".format(epoch))
                pre_acc = acc
                early_stop_patience = 0
            else:
                early_stop_patience += 1
                if early_stop_patience > cfg.GLOBAL.EARLY_STOP_PATIENCE:
                    logger.info('acc exceeds times without improvement, stop training earily')
                    sys.exit(1)
    # destroy process
    if world_size > 1 and rank == 0:
        dist.destroy_process_group()


if __name__ == "__main__":
    args = get_args()   # 读取入参
    cfg = yaml_parser(args.yaml) # 解析配置文件
    experiment_dir, checkpoints_dir, tensorboard_dir = init_setting(cfg)
    train(cfg)  # 开始训练的主