import sys
import os 
import argparse

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.myClassDataset import MyDataset
from model.build_model import get_model
from common.util import yaml_parser
from common.heatmap_view import save_heatmap
from common.feature_view import save_feature


def cal_pr_acc(model, args, cfg):
    exp_dir = '/'.join(args.model_path.split('/')[:3])
    model.eval()
    data_transform = transforms.Compose([
                transforms.Resize((cfg.TRAIN.IMG_SIZE, cfg.TRAIN.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    
    root_path = cfg.GLOBAL.VAL_DIR
    classes = os.listdir(root_path)
    classes.sort()
    pr_matrix = [[0] * (len(classes)) for _ in range(len(classes))]
    cur_recall_list = []
    cur_precision_list = []
    
    for single_dir_index in range(len(classes)):
        single_dir_name = classes[single_dir_index]
        single_dir = os.path.join(root_path, single_dir_name)
        for img in os.listdir(single_dir):
            img_path = os.path.join(single_dir, img)
            img = Image.open(img_path)
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)
            with torch.no_grad():
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)  # 进行softmax，将概率映射到0-1范围
                predict_cla = torch.argmax(predict).numpy() # 取出最大概率的那个类别
                str_res = "gt : {}   predict : {}   prob: {:.3}".format(single_dir_name, str(predict_cla), predict[predict_cla].numpy())
                print(str_res)
                pr_matrix[single_dir_index][int(predict_cla)] += 1
        cur_recall = pr_matrix[single_dir_index][single_dir_index] / (sum(pr_matrix[single_dir_index])) # 先计算召回
        # print("cur_recall : %.2f%%"  % (cur_recall * 100))
        cur_recall_list.append(cur_recall)
                
    for single_dir_index in range(len(classes)):
        single_dir_name = classes[single_dir_index]
        single_dir = os.path.join(root_path, single_dir_name)
        count = 1e-4
        for i in range(len(pr_matrix)):
            count += pr_matrix[i][single_dir_index]
        cur_precision = pr_matrix[single_dir_index][single_dir_index] / count   # 计算精确率
        cur_precision_list.append(cur_precision)
    print("样本总数 : %d" % np.sum(pr_matrix))
    print("整体样本的召回率 : %.2f%%" % (np.mean(cur_recall_list) * 100))
    print("整体样本的精确率 : %.2f%%" % (np.mean(cur_precision_list) * 100))
    print("整体样本的准确率 : %.2f%%" % ((np.trace(pr_matrix) / np.sum(pr_matrix)) * 100))
    plt.clf()
    fig, ax = plt.subplots(figsize=(11, 11))
    ax = sns.heatmap(pr_matrix, xticklabels=classes, yticklabels=classes, center=30, cmap="RdBu_r", linewidths=0.5,linecolor="grey", annot=True, fmt="d")
    ax.set_title('total sample : %d\nrecall : %.2f%%\nprecise : %.2f%%\nacc : %.2f%%' % (np.sum(pr_matrix), np.mean(cur_recall_list) * 100, np.mean(cur_precision_list) * 100, np.trace(pr_matrix) / np.sum(pr_matrix) * 100), 
                fontsize=8)
    ax.set_xticklabels(classes, rotation=90,fontsize=15)
    ax.set_yticklabels(classes, rotation=0, fontsize=15)

    figure = ax.get_figure()
    figure.savefig(os.path.join(exp_dir, 'pr_matrix_heatmap.jpg'), dpi=300)
    plt.close()


def evaluate(model, data_loader, device):
    with torch.no_grad():
        model.eval()
        
        total_num = len(data_loader.dataset)
        data_loader = tqdm(data_loader, file=sys.stdout)
        sum_num = torch.zeros(1).to(device)

        for _, data in enumerate(data_loader):
            images, labels = data
            pred = model(images.to(device))
            pred = torch.max(pred, dim=1)[1]
            sum_num += torch.eq(pred, labels.to(device)).sum()  # 模型输出和gt作对比

    return sum_num.item() / total_num


def get_args():
    parser = argparse.ArgumentParser(description='val')
    parser.add_argument('--yaml', default='config/train.yaml', type=str, help='output model name')
    parser.add_argument('--model_path', default='train_log/2022-1018_1411/checkpoints/best.pth', type=str, help='output model name')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    cfg = yaml_parser(args.yaml)
    num_classes = len(os.listdir(cfg.GLOBAL.VAL_DIR))

    if cfg.GLOBAL.DEVICE == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif cfg.GLOBAL.DEVICE:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GLOBAL.DEVICE
        assert torch.cuda.is_available()
    cuda = cfg.GLOBAL.DEVICE != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    net = get_model(cfg, num_classes)
    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint['state_dict_backbone'])
    net.to(device)

    val_dataset = MyDataset(cfg=cfg, mode='val')
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.TRAIN.BATCHSIZE_PER_CARD,
                            shuffle=False,
                            num_workers=cfg.TRAIN.NUM_WORKERS,
                            drop_last=cfg.TRAIN.DROP_LAST,
                            collate_fn=val_dataset.collate_fn)
    # 评估验证集
    acc = evaluate(net, val_loader, device)
    print("acc", acc)
    
    # 计算acc、 recall、 precision
    cal_pr_acc(net, args, cfg)
    
    imgs_path = "data_dir/test/dogs/dog.0.jpg"
    save_heatmap(net, args, cfg, imgs_path, device) # 热力图可视化
    save_feature(net, args, cfg, imgs_path, device) # 特征可视化
