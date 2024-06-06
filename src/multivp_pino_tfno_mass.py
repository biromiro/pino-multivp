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
from modulus.sym.eq.pdes.diffusion import Diffusion

from momentum_conservation import MomentumConservation
from mass_conservation import MassConservation

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


@hydra.main(version_base="1.3", config_path="conf", config_name="config_pino_tfno_mass.yaml")
def main(cfg: DictConfig):

    # CUDA support
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    LaunchLogger.initialize()

    # Momentum Conservation Eqs
    mom = MomentumConservation()
    mom_nodes = mom.make_nodes()
    
    # Mass Conservation Eqs
    mass = MassConservation()
    mass_nodes = mass.make_nodes()
    
    DX_UN  =   6.96000e+10  # length unit factor (to cm)
    VX_UN  =   1e+05 # speed unit factor (to cm/s)
    TE_UN  =   1.00000e+06  # temperature unit factor (to K)
        
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

    attenuation_factor_mom = 1
    attenuation_factor_mass = 1
    
    for epoch in range(cfg.max_epochs):
        # wrap epoch in launch logger for console logs
        if epoch > 0 and epoch < cfg.epoch_threshold_upper and epoch % 25 == 0:
            attenuation_factor_mom = attenuation_factor_mom * 1.4 if (cfg.phy_att_mom_max > cfg.phy_wt_mom * attenuation_factor_mom) else attenuation_factor_mom
            attenuation_factor_mass = attenuation_factor_mass * 1.4 if (cfg.phy_att_mass_max > cfg.phy_wt_mass * attenuation_factor_mass) else attenuation_factor_mass
            print(f'New attenuation factor momentum: {attenuation_factor_mom}')
            print(f'New attenuation factor mass: {attenuation_factor_mass}')
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg.optimizer_params.lr * 0.1
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.optimizer_params.gamma)
        
        
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(dataloader),
            epoch_alert_freq=10,
        ) as log:
            for data in dataloader:
                class OptimizationClosure:
                    def __init__(self) -> None:
                        self.loss_data = torch.nan
                        self.loss_pde = torch.nan
                        self.loss_pde_mass = torch.nan
                        self.loss_pde_mom = torch.nan
                    
                    def closure(self):
                        optimizer.zero_grad()
                        invar = data[0]
                        outvar = data[1]
                        #print(invar.shape, outvar.shape, invar[:, 0].unsqueeze(dim=1).shape)
                        # Compute forward pass
                        out = model(invar)
                        
                        if epoch > cfg.epoch_threshold:
                            invar_denorm = denormalize(invar, norm_info[0])
                            out_denorm = denormalize(out, norm_info[1])
                            
                            invar_denorm[:, 0, :] *= DX_UN
                            invar_denorm[:, 1, :] *= DX_UN
                            invar_denorm[:, 3, :] /= DX_UN

                            out_denorm[:, 1, :] *= VX_UN
                            out_denorm[:, 2, :] *= TE_UN
                            
                            idx = 248
                            
                            R, L, B, a_a0, alpha = (
                                invar_denorm[:, 0, idx:],
                                invar_denorm[:, 1, idx:],
                                invar_denorm[:, 2, idx:],
                                invar_denorm[:, 3, idx:],
                                invar_denorm[:, 4, idx:]
                            )
                            
                            
                            n, v, T = (
                                out_denorm[:, 0, idx:],
                                out_denorm[:, 1, idx:],
                                out_denorm[:, 2, idx:],
                            )
                            
                            L_coords = L[0, :]

                            dL_diff = L_coords[1:] - L_coords[:-1]
                                
                            dfL = torch.zeros_like(L_coords)
                            dfL[0] = dL_diff[0]
                            dfL[-1] = dL_diff[-1]

                            dfL[1:-1] = (dL_diff[:-1] + dL_diff[1:]) / 2

                            # Compute gradients using finite difference
                            # v__L, n__L, T__L, v__L, v__L__L
                            B__L = dx(invar_denorm[:, :, idx:], dx=dfL, channel=2, order=3, padding="zeros")
                            n__L = dx(out_denorm[:, :, idx:], dx=dfL, channel=0, order=3, padding="zeros")
                            v__L = dx(out_denorm[:, :, idx:], dx=dfL, channel=1, order=3, padding="zeros")
                            T__L = dx(out_denorm[:, :, idx:], dx=dfL, channel=2, order=3, padding="zeros")
                            v__L__L = ddx(out_denorm[:, :, idx:], dx=dfL, channel=1, order=3, padding="zeros")
                            
                            # a_a0, v__L, T, n__L, n, T__L, n, v, v__L, v__L__L, cos_alpha, R
                            pde_out_mom = mom_nodes[0].evaluate(
                                {
                                    "R": R,
                                    "a_a0": a_a0,
                                    "cos_alpha": torch.cos(alpha),
                                    "n": n,
                                    "v": v,
                                    "T": T,
                                    "n__L": n__L,
                                    "v__L": v__L,
                                    "v__L__L": v__L__L,
                                    "T__L": T__L,
                                }
                            )


                            pde_out_arr_mom = pde_out_mom["mom_term"]
                            pde_out_arr_mom = F.pad(
                                pde_out_arr_mom[:, :, 2:-2], [2, 2, 2, 2], "constant", 0
                            )
                            self.loss_pde_mom = F.l1_loss(pde_out_arr_mom, torch.zeros_like(pde_out_arr_mom))
                            
                            pde_out_arr_mass = torch.std(n*v/B, dim=1).mean().sqrt()
                            self.loss_pde_mass = F.l1_loss(pde_out_arr_mass, torch.zeros_like(pde_out_arr_mass))
                            
                            self.loss_pde = attenuation_factor_mom * cfg.phy_wt_mom * self.loss_pde_mom \
                                    + attenuation_factor_mass * cfg.phy_wt_mass * self.loss_pde_mass
                    
                        # Compute data loss
                        self.loss_data = F.mse_loss(outvar, out)
                        # Compute total loss          
                        loss = (self.loss_data + self.loss_pde) if epoch > cfg.epoch_threshold else self.loss_data
                        
                        # Backward pass and optimizer and learning rate update
                        loss.backward()
                        return loss
                
                closure_obj = OptimizationClosure()
                optimizer.step(closure=closure_obj.closure)
                log.log_minibatch(
                    {
                        "loss_data": closure_obj.loss_data.detach(), 
                        "loss_pde": closure_obj.loss_pde.detach() if epoch > cfg.epoch_threshold else torch.nan,
                        "loss_pde_mom": closure_obj.loss_pde_mom.detach() if epoch > cfg.epoch_threshold else torch.nan,
                        "loss_pde_mass": closure_obj.loss_pde_mass.detach() if epoch > cfg.epoch_threshold else torch.nan,
                    }
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
