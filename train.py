import numpy as np
import pytorch_lightning as pl

from models.nerf_model import NeRFModel
from data.datamodule import NeRFDataModule


data_path = 'tiny_nerf_data.npz'
data = np.load(data_path)
img_H, img_W = data['images'].shape[1:3]
img_focal = data['focal']
datamodule = NeRFDataModule(data_path)

# 初始化NeRFModel
model = NeRFModel(img_H=img_H, img_W=img_W, img_focal=img_focal, N_samples=64)

# 设置训练参数
N_iters = 100
trainer = pl.Trainer(max_epochs=N_iters)

# 开始训练
trainer.fit(model, datamodule=datamodule)
