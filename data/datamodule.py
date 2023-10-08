import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

# 假设你已经在datasets.py中定义了RandomCamerasDataset
from .datasets import TinyNerfDataset

class NeRFDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_path = data_path
        data = np.load(self.data_path)
        self.images = data['images']
        self.poses = data['poses']
        self.focal = data['focal']
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 初始化数据集
        self.dataset = TinyNerfDataset(self.data_path)

    def setup(self, stage=None):
        # 划分数据集为训练集和验证集
        train_length = int(0.8 * len(self.dataset))
        val_length = len(self.dataset) - train_length
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_length, val_length])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        # 这里，我们只使用一个图像和姿势作为验证数据
        testimg_tensor = torch.from_numpy(self.images[101]).permute(2, 0, 1).float()
        testpose_tensor = torch.from_numpy(self.poses[101]).float()
        val_dataset = [(testimg_tensor, testpose_tensor)]
        return DataLoader(val_dataset, batch_size=1)
