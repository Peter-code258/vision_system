# vision_system/models/dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, data_dir, labels_file, transform=None):
        """
        data_dir: 图像所在文件夹路径
        labels_file: 标签文件路径，通常是包含图像路径和标签的csv/json文件
        transform: 预处理操作，使用 torchvision.transforms
        """
        self.data_dir = data_dir
        self.labels_file = labels_file
        self.transform = transform

        # 加载标签数据（假设 labels_file 是一个 CSV）
        with open(labels_file, 'r') as file:
            self.labels = [line.strip().split(',') for line in file.readlines()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.labels[idx][0])
        label = int(self.labels[idx][1])  # 例如 label 存储的是数字类标签

        # 读取图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB

        if self.transform:
            img = self.transform(img)

        return img, label
