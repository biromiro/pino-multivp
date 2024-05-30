# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from hydra.utils import to_absolute_path
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from tfno import TFNO
from modulus.launch.logging import LaunchLogger
from modulus.launch.utils.checkpoint import save_checkpoint
#from modulus.sym.eq.pdes.diffusion import Diffusion

#from utils import HDF5MapStyleDataset
from ops import dx, ddx

class TensorRobustScaler:
    def __init__(self):
        self.median = None
        self.iqr = None

    def fit(self, X):
        X = X.view(-1)
        self.median = torch.quantile(X, 0.5, dim=-1)
        q1 = torch.quantile(X, 0.25, dim=-1)
        q3 = torch.quantile(X, 0.75, dim=-1)
        self.iqr = q3 - q1

    def transform(self, X):
        return (X - self.median) / self.iqr

    def inverse_transform(self, X):
        return (X * self.iqr) + self.median

def denormalize(X_normalized, normalization_info):
    X_denormalized = X_normalized.clone()
    
    for var, info in normalization_info.items():
        if info["method"] == "standardization":
            mean = info["mean"]
            std = info["std"]
            X_denormalized[:, var, :] = (X_denormalized[:, var, :] * std) + mean
        if info["method"] == "log_standardization":
            mean = info["mean"]
            std = info["std"]
            X_denormalized[:, var, :] = torch.expm1((X_denormalized[:, var, :] * std) + mean)
        elif info["method"] == "log_robust_scaling":
            scaler = info["scaler"]
            X_denormalized[:, var, :] = torch.expm1(scaler.inverse_transform(X_denormalized[:, var, :]))
    
    return X_denormalized

def validation_step(model, dataset, norm_info, epoch):
    """Validation Step"""
    model.eval()

    with torch.no_grad():
        invar, outvar = dataset.tensors
        out = model(invar)
        loss_epoch = F.mse_loss(outvar, out)

        if epoch % 25 != 0:
            return loss_epoch
        
        # convert data to numpy
        outvar = denormalize(outvar, norm_info[1]).detach().cpu().numpy()
        predvar = denormalize(out, norm_info[1]).detach().cpu().numpy()
        # smooth predictions
        predvar_smooth = predvar.copy()

        for i in range(predvar_smooth.shape[0]):
            for j in range(predvar_smooth.shape[1]):
                predvar_smooth[i, j, :] = pd.Series(predvar_smooth[i, j, :]).rolling(window=8, min_periods=1, win_type='hamming').mean()
        # plotting
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))

        output_names = ['n', 'v', 'T']
        lims = [[50, 500000000], [0, 800], [0, 5]]

        for i in range(3):
            axs[i, 0].plot(outvar[:, i, :].T)
            axs[i, 0].set_yscale('log') if i == 0 else None
            axs[i, 0].set_ylim(lims[i])
            axs[i, 1].plot(predvar[:, i, :].T)
            axs[i, 1].set_yscale('log') if i == 0 else None
            axs[i, 1].set_ylim(lims[i])
            axs[i, 2].plot(predvar_smooth[:, i, :].T)
            axs[i, 2].set_yscale('log') if i == 0 else None
            axs[i, 2].set_ylim(lims[i])
            
            axs[i, 0].set_title(f'{output_names[i]} true')
            axs[i, 1].set_title(f'{output_names[i]} predicted')
            axs[i, 2].set_title(f'{output_names[i]} smooth predicted')

        fig.savefig(f"results_{epoch}.png")
        plt.close()
        
        return loss_epoch


@hydra.main(version_base="1.3", config_path="conf", config_name="config_tfno.yaml")
def main(cfg: DictConfig):

    # CUDA support
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    LaunchLogger.initialize()

    # Use Diffusion equation for the Darcy PDE
    #darcy = Diffusion(T="u", time=False, dim=2, D="k", Q=1.0 * 4.49996e00 * 3.88433e-03)
    #darcy_node = darcy.make_nodes()
    X_train_normalized = torch.load(to_absolute_path('./datasets/multivp/X_train_normalized.pt')).to(torch.float32).to(device)
    y_train_normalized = torch.load(to_absolute_path('./datasets/multivp/y_train_normalized.pt')).to(torch.float32).to(device)
    X_val_normalized = torch.load(to_absolute_path('./datasets/multivp/X_val_normalized.pt')).to(torch.float32).to(device)
    y_val_normalized = torch.load(to_absolute_path('./datasets/multivp/y_val_normalized.pt')).to(torch.float32).to(device)
    # load normalization_info_inputs.pt
    normalization_info_inputs = torch.load(
        to_absolute_path('./datasets/multivp/normalization_info_inputs.pt'), map_location=device)
    normalization_info_outputs = torch.load(
        to_absolute_path('./datasets/multivp/normalization_info_outputs.pt'), map_location=device)
    
    norm_info = (normalization_info_inputs, normalization_info_outputs)
    
    dataset = TensorDataset(X_train_normalized, y_train_normalized)
    validation_dataset = TensorDataset(X_val_normalized, y_val_normalized)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    #validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    model = TFNO(
        in_channels=cfg.model.tfno.in_channels,
        out_channels=cfg.model.tfno.out_channels,
        decoder_layers=cfg.model.tfno.decoder_layers,
        decoder_layer_size=cfg.model.tfno.decoder_layer_size,
        dimension=cfg.model.tfno.dimension,
        latent_channels=cfg.model.tfno.latent_channels,
        num_fno_layers=cfg.model.tfno.num_fno_layers,
        num_fno_modes=cfg.model.tfno.num_fno_modes,
        padding=cfg.model.tfno.padding,
        rank=cfg.model.tfno.rank,
        factorization=cfg.model.tfno.factorization,
        fixed_rank_modes=cfg.model.tfno.fixed_rank_modes,
        decomposition_kwargs=cfg.model.tfno.decomposition_kwargs,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        betas=cfg.optimizer_params.betas,
        lr=cfg.optimizer_params.lr,
        weight_decay=cfg.optimizer_params.weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.optimizer_params.gamma)

    for epoch in range(cfg.max_epochs):
        # wrap epoch in launch logger for console logs
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(dataloader),
            epoch_alert_freq=10,
        ) as log:
            for data in dataloader:
                optimizer.zero_grad()
                invar = data[0]
                outvar = data[1]
                #print(invar.shape, outvar.shape, invar[:, 0].unsqueeze(dim=1).shape)
                # Compute forward pass
                out = model(invar)

                """dxf = 1.0 / out.shape[-2]
                dyf = 1.0 / out.shape[-1]

                # Compute gradients using finite difference
                sol_x = dx(out, dx=dxf, channel=0, dim=1, order=1, padding="zeros")
                sol_y = dx(out, dx=dyf, channel=0, dim=0, order=1, padding="zeros")
                sol_x_x = ddx(out, dx=dxf, channel=0, dim=1, order=1, padding="zeros")
                sol_y_y = ddx(out, dx=dyf, channel=0, dim=0, order=1, padding="zeros")

                k_x = dx(invar, dx=dxf, channel=0, dim=1, order=1, padding="zeros")
                k_y = dx(invar, dx=dxf, channel=0, dim=0, order=1, padding="zeros")

                k, _, _ = (
                    invar[:, 0],
                    invar[:, 1],
                    invar[:, 2],
                )

                pde_out = darcy_node[0].evaluate(
                    {
                        "u__x": sol_x,
                        "u__y": sol_y,
                        "u__x__x": sol_x_x,
                        "u__y__y": sol_y_y,
                        "k": k,
                        "k__x": k_x,
                        "k__y": k_y,
                    }
                )

                pde_out_arr = pde_out["diffusion_u"]
                pde_out_arr = F.pad(
                    pde_out_arr[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0
                )
                loss_pde = F.l1_loss(pde_out_arr, torch.zeros_like(pde_out_arr))"""

                # Compute data loss
                loss_data = F.mse_loss(outvar, out)

                # Compute total loss
                loss = loss_data# + 1 / 240 * cfg.phy_wt * loss_pde

                # Backward pass and optimizer and learning rate update
                loss.backward()
                optimizer.step()
                log.log_minibatch(
                    {"loss_data": loss_data.detach()}#, "loss_pde": loss_pde.detach()}
                )

            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
            scheduler.step()

        with LaunchLogger("valid", epoch=epoch) as log:
            error = validation_step(model, validation_dataset, norm_info, epoch)
            log.log_epoch({"Validation error": error})
            pass

        save_checkpoint(
            "./checkpoints",
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )


if __name__ == "__main__":
    main()
