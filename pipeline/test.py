import os
import argparse

import torch
from PIL import Image
from torchvision import transforms

from model.build_model import get_model
from common.util import yaml_parser


def inference_single_img(cfg, net, img_path, device):
    data_transform = transforms.Compose([
                transforms.Resize((cfg.TRAIN.IMG_SIZE, cfg.TRAIN.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    
    root_path = cfg.GLOBAL.TEST_DIR
    classes = os.listdir(root_path)
    classes.sort()
    class_indices = dict((k, v) for k, v in enumerate(classes))
    
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    # read class_indict
    net.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(net(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    print(img_path)
    res = "class: {}   prob: {:.3}".format(class_indices[int(predict_cla)], predict[predict_cla].numpy())
    print(res)
    return class_indices[int(predict_cla)], predict[predict_cla].numpy()


def inference_img_list(cfg, net, imgs_dir, device, val=False):      #  使用val判断使用哪种模式
    imgs_list = os.listdir(imgs_dir)
    if not val:
        for img_name in imgs_list:
            img_path = os.path.join(imgs_dir, img_name)
            _, _ = inference_single_img(cfg, net, img_path, device)
        return "done"
    else:
        total_num = 0
        right = 0
        for img_name in imgs_list:
            total_num += 1
            img_path = os.path.join(imgs_dir, img_name)
            res_cls, res_prob = inference_single_img(cfg, net, img_path, device)
            if res_cls == img_path.split('/')[-2]:
                right += 1
        return right / total_num


def test(cfg, net, val_root_path, device):
    cls_names = os.listdir(val_root_path)
    result = {}
    for cls_name in cls_names:
        imgs_dir = os.path.join(val_root_path, cls_name)
        acc = inference_img_list(cfg, net, imgs_dir, device, val=True)
        result[cls_name] = acc
    print(result)


def get_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--yaml', default='config/train.yaml', type=str, help='output model name')
    parser.add_argument('--model_path', default='train_log/2022-1018_1411/checkpoints/best.pth', type=str, help='output model name')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    cfg = yaml_parser(args.yaml)
    num_classes = len(os.listdir(cfg.GLOBAL.TEST_DIR))
    if cfg.GLOBAL.DEVICE == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif cfg.GLOBAL.DEVICE:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GLOBAL.DEVICE
        assert torch.cuda.is_available()
    cuda = cfg.GLOBAL.DEVICE != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    net = get_model(cfg, num_classes)   # 创建模型
    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint['state_dict_backbone'])
    net.to(device)

    # 推理单张图片
    img_path = "data_dir/test/horses/OIP-BWaRPN9UdS1zHXMqIvRE9gHaE8.jpeg"
    res_cls, res_prob = inference_single_img(cfg, net, img_path, device)

    # 推理一个文件夹下的图片
    imgs_dir = "data_dir/test/cats"
    _ = inference_img_list(cfg, net, imgs_dir, device)

    # 评估整个验证集
    val_root_path = "data_dir/test"
    res = test(cfg, net, val_root_path, device)
