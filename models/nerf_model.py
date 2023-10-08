from typing import Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from utils.ray_utils import get_rays, render_rays


class NeRFModel(pl.LightningModule):
    def __init__(self,
                 D: int = 8,
                 width: int = 256,
                 L_embed: int = 6,
                 N_samples: int = 64,
                 img_H: Optional[int] = None,
                 img_W: Optional[int] = None,
                 img_focal: Optional[int] = None):
        super(NeRFModel, self).__init__()
        self.img_H = img_H
        self.img_W = img_W
        self.img_focal = img_focal
        self.L_embed = L_embed
        self.N_samples = N_samples
        self.embed_fn = self.posenc

        layers = []
        for i in range(D):
            if i == 0:
                layers.append(nn.Linear(3 + 3 * 2 * L_embed, width))
            elif i == 4:
                layers.append(nn.Linear(3 + 3 * 2 * L_embed + width, width))
            else:
                layers.append(nn.Sequential(*[nn.Linear(width, width), nn.ReLU()]))
        layers.append(nn.Linear(width, 4))
        self.layers = nn.ModuleList(layers)

    def posenc(self, x):
        rets = [x]
        for i in range(self.L_embed):
            for fn in [torch.sin, torch.cos]:
                rets.append(fn(2.**i * x))
        return torch.cat(rets, -1)

    def forward(self, x):
        x = self.embed_fn(x)
        x0 = x
        for i, layer in enumerate(self.layers):
            if i == 4:
                x = torch.cat([x0, x], -1)
            x = layer(x)
        return x

    def training_step(self, batch, batch_idx):
        target, pose = batch
        rays_o, rays_d = get_rays(self.img_H, self.img_W, self.img_focal, pose)
        rgb, depth, acc = render_rays(
            self, rays_o, rays_d, near=2., far=6., N_samples=self.N_samples, rand=True)
        loss = torch.mean((rgb - target) ** 2)
        return loss

    def validation_step(self, batch, batch_idx):
        testimg, testpose = batch
        rays_o, rays_d = get_rays(
            self.img_H, self.img_W, self.img_focal, testpose)
        rgb, depth, acc = render_rays(
            self, rays_o, rays_d, near=2., far=6., N_samples=self.N_samples)
        testimg = testimg.squeeze().permute(1, 2, 0)
        loss = torch.mean((rgb - testimg) ** 2)
        psnr = -10. * torch.log10(loss)
        self.log('val_psnr', psnr)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-4)
