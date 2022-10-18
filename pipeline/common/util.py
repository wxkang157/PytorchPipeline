import argparse
import datetime
import math
import random
import glob
import os

from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from shutil import copyfile

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


def yaml_parser(yaml_path):
    with open(yaml_path, 'r') as file:
        opt = argparse.Namespace(**yaml.load(file.read(), Loader=yaml.FullLoader))
    opt.GLOBAL = argparse.Namespace(**opt.GLOBAL)
    opt.TRAIN = argparse.Namespace(**opt.TRAIN)
    opt.MODEL = argparse.Namespace(**opt.MODEL)
    opt.CRITERION = argparse.Namespace(**opt.CRITERION)
    opt.OPTIMIZER = argparse.Namespace(**opt.OPTIMIZER)

    return opt


def init_setting(cfg):
    timestr = str(datetime.datetime.now().strftime('%Y-%m%d_%H%M'))
    experiment_dir = Path(cfg.GLOBAL.SAVE_RESULT_DIR)
    experiment_dir.mkdir(exist_ok=True) # 保存实验结果的总目录
    experiment_dir = experiment_dir.joinpath(timestr)
    experiment_dir.mkdir(exist_ok=True) # 每次实验的根目录
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)    # 保存模型的目录
    tensorboard_dir = experiment_dir.joinpath('tensorboard/')
    tensorboard_dir.mkdir(exist_ok=True)    # 保存日志的目录
    setting_dir = experiment_dir.joinpath('setting/')
    setting_dir.mkdir(exist_ok=True)  # 保存日志的目录

    copyfile('data/myClassDataset.py', str(setting_dir) + '/myClassDataset.py')
    copyfile('config/my.yaml', str(setting_dir) + '/my.yaml')
    copyfile('loss/build_loss.py', str(setting_dir) + '/build_loss.py')
    copyfile('model/build_model.py', str(setting_dir) + '/build_model.py')
    copyfile('train.py', str(setting_dir) + '/train.py')
    copyfile('test.py', str(setting_dir) + '/test.py')
    copyfile('val.py', str(setting_dir) + '/val.py')

    return experiment_dir, checkpoints_dir, tensorboard_dir


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = f'Total : {str(total_num / 1000 ** 2)} M, Trainable: {str(trainable_num / 1000 ** 2)} M'
    return info


def read_split_data(cfg, mode):
    # 遍历文件夹，一个文件夹对应一个类别
    classes = [cla for cla in os.listdir(cfg.GLOBAL.TRAIN_DIR)]
    # 排序，保证顺序一致
    classes.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(classes))

    images_path = []
    images_label = []
    every_class_num = []
    
    if mode == 'train':
        for cla in classes:
            train_cla_path = os.path.join(cfg.GLOBAL.TRAIN_DIR, cla)
            images = [os.path.join(cfg.GLOBAL.TRAIN_DIR, cla, i) for i in os.listdir(train_cla_path)]
            image_class = class_indices[cla]
            every_class_num.append(len(images)) # 记录每个类别下的图片个数
            for img_path in images:
                images_path.append(img_path)
                images_label.append(image_class)
    elif mode == 'val':
        for cla in classes:
            val_cal_path = os.path.join(cfg.GLOBAL.VAL_DIR, cla)
            # 遍历获取supported支持的所有文件路径
            images = [os.path.join(cfg.GLOBAL.VAL_DIR, cla, i) for i in os.listdir(val_cal_path)]
            # 获取该类别对应的索引
            image_class = class_indices[cla]
            # 记录该类别的样本数量
            every_class_num.append(len(images))
            for img_path in images:
                images_path.append(img_path)
                images_label.append(image_class)

    return every_class_num, images_path, images_label


def plot_image(num_classes, every_class_num, experiment_dir, mode):

    plt.bar(range(len(num_classes)), every_class_num, align='center')
    # 将横坐标0,1,2,3,4替换为相应的类别名称
    plt.xticks(range(len(num_classes)), num_classes)
    # 在柱状图上添加数值标签
    for i, v in enumerate(every_class_num):
        plt.text(x=i, y=v + 5, s=str(v), ha='center')
    # 设置x坐标
    if mode == 'train':
        plt.xlabel('train image class')
    elif mode == 'val':
        plt.xlabel('val image class')
    # 设置y坐标
    plt.ylabel('number of images')
    # 设置柱状图的标题
    plt.title('class distribution')
    if mode == 'train':
        plt.savefig(os.path.join(experiment_dir, 'train_dataset.png'))
        plt.close()
    elif mode == 'val':
        plt.savefig(os.path.join(experiment_dir, 'val_dataset.png'))
        plt.close()


def view_dataset(experiment_dir, cfg):
    img_list = glob.glob(cfg.GLOBAL.TRAIN_DIR + '/*/*.jpg')
    train_num_classes = os.listdir(cfg.GLOBAL.TRAIN_DIR)
    train_num_classes.sort()    # 排序，为了每次排序的结果一致
    val_num_classes = os.listdir(cfg.GLOBAL.VAL_DIR)
    val_num_classes.sort()
    random.shuffle(img_list)
    img_list = img_list[:9]
    transform_list = [
        transforms.Resize((cfg.TRAIN.IMG_SIZE, cfg.TRAIN.IMG_SIZE)),
        transforms.RandomCrop((cfg.TRAIN.IMG_SIZE, cfg.TRAIN.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=cfg.TRAIN.TRANSFORMS_BRIGHTNESS,
                               contrast=cfg.TRAIN.TRANSFORMS_CONTRAST,
                               saturation=cfg.TRAIN.TRANSFORMS_SATURATION,
                               hue=cfg.TRAIN.TRANSFORMS_HUE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transform = transforms.Compose(transform_list)
    test_img_list = []
    for pic_path in img_list:
        test_img = Image.open(pic_path)
        test_img_list.append(test_img)
    nrows = 3
    ncols = 3
    figsize = (8, 8)
    _, figs = plt.subplots(nrows, ncols, figsize=figsize)
    for i in range(nrows):
        for j in range(ncols):
            img = transform(test_img_list[i + j])   # 选取3x3=9张图片进行数据增强可视化
            img = img.numpy().transpose((1, 2, 0))
            img = np.clip(img, 0, 1)
            figs[i][j].imshow(img)
            figs[i][j].axes.get_xaxis().set_visible(False)
            figs[i][j].axes.get_yaxis().set_visible(False)
    plt.savefig(os.path.join(experiment_dir, 'dataset_aug.png'))
    plt.close()

    train_every_class_num, _, _ = read_split_data(cfg, 'train') # 读取数据集
    val_every_class_num, _, _ = read_split_data(cfg, 'val')
    plot_image(train_num_classes, train_every_class_num, experiment_dir, 'train')   # 画图
    plot_image(val_num_classes, val_every_class_num, experiment_dir, 'val')


def build_scheduler(optimizer, cfg):
    epochs = cfg.GLOBAL.EPOCH_NUM
    if cfg.OPTIMIZER.LR_NAME == 'linear_lr':
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - cfg.OPTIMIZER.LR_LRF) + cfg.OPTIMIZER.LR_LRF 
    elif cfg.OPTIMIZER.LR_NAME == 'cosine_lr':
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (1.0 - cfg.OPTIMIZER.LR_LRF) + cfg.OPTIMIZER.LR_LRF

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    return scheduler


def build_optimizer(model, cfg, logger):
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if cfg.OPTIMIZER.NAME == "Adam":
        optimizer = Adam(g0, lr=cfg.OPTIMIZER.LEARNING_RATE, 
                               betas=[cfg.OPTIMIZER.BETA1, cfg.OPTIMIZER.BETA2])
        optimizer.add_param_group({'params': g1, 'weight_decay': cfg.OPTIMIZER.WEIGHT_DECAY})  # add g1 with weight_decay
        optimizer.add_param_group({'params': g2})  # add g2 (biases)
    elif cfg.OPTIMIZER.NAME == "SGD":
        optimizer = SGD(g0, lr=cfg.OPTIMIZER.LEARNING_RATE, 
                        momentum=cfg.OPTIMIZER.MOMENTUM, 
                        nesterov=cfg.OPTIMIZER.NESTEROV)
        optimizer.add_param_group({'params': g1, 'weight_decay': cfg.OPTIMIZER.WEIGHT_DECAY})  # add g1 with weight_decay
        optimizer.add_param_group({'params': g2})  # add g2 (biases)
    logger.info(f"{'optimizer:'} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
    del g0, g1, g2

    return optimizer

