import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class NeRFDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): 数据的目录路径。
            transform (callable, optional): 可选的变换操作。
        """
        self.data_dir = data_dir
        self.transform = transform
        self.data_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_files[idx])
        # 加载数据，例如使用 PIL.Image.open(data_path)
        data = ...
        
        if self.transform:
            data = self.transform(data)
        
        return data

class TinyNerfDataset(Dataset):
    def __init__(self, data_path: str):
        super(TinyNerfDataset, self).__init__()
        
        # 加载数据
        data = np.load(data_path)
        self.images = data['images']
        self.poses = data['poses']
        self.focal = data['focal']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        pose = self.poses[idx]
        
        # 将numpy数组转换为torch张量
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # CxHxW
        pose = torch.from_numpy(pose).float()
        
        return image, pose


