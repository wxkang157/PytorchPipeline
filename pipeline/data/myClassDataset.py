import sys

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from common.util import read_split_data


class MyDataset(data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        every_class_num, self.images_path, self.images_label = read_split_data(self.cfg, self.mode)

    def __getitem__(self, index):
        img = Image.open(self.images_path[index])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[index]))
        data = self.preprocess(img)
        label = self.images_label[int(index)]

        return data, label

    def __len__(self):
        return len(self.images_path)

    def preprocess(self, data):
        # Here are just some of the operations, for more operations, please visit:
        # https://pytorch.org/vision/stable/transforms.html#compositions-of-transforms

        if self.mode == "train":
            transform_list = [
                transforms.Resize((self.cfg.TRAIN.IMG_SIZE, self.cfg.TRAIN.IMG_SIZE)),
                transforms.RandomCrop((self.cfg.TRAIN.IMG_SIZE, self.cfg.TRAIN.IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=self.cfg.TRAIN.TRANSFORMS_BRIGHTNESS,
                                       contrast=self.cfg.TRAIN.TRANSFORMS_CONTRAST,
                                       saturation=self.cfg.TRAIN.TRANSFORMS_SATURATION,
                                       hue=self.cfg.TRAIN.TRANSFORMS_HUE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ]
        elif self.mode == "val" or self.mode == "test" or self.mode == "inference":
            transform_list = [
                transforms.Resize((self.cfg.TRAIN.IMG_SIZE, self.cfg.TRAIN.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ]
        else:
            print("mode only support [train, val, test, inference]!")
            sys.exit(0)

        transform = transforms.Compose(transform_list)
        data = transform(data)
        return data

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
