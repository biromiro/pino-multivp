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

import torch
import torch.nn.functional as F

def dx(inpt, dx, channel, order=1, padding="zeros"):
    "Compute first order numerical derivatives of input tensor with variable spacing"

    # Extract the specified channel
    var = inpt[:, channel : channel + 1, :]
    
    # Ensure dx is a tensor
    if not isinstance(dx, torch.Tensor):
        dx = torch.tensor(dx, device=inpt.device)
    
    # Get the filter
    if order == 1:
        ddx1D = torch.Tensor([-0.5, 0.0, 0.5]).to(inpt.device)
    elif order == 3:
        ddx1D = torch.Tensor([-1.0 / 60.0, 3.0 / 20.0, -3.0 / 4.0, 0.0, 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0]).to(inpt.device)

    ddx1D = ddx1D.view(1, 1, -1)  # reshape to [1, 1, filter_length]

    # Apply padding
    pad_size = (ddx1D.shape[2] - 1) // 2
    if padding == "zeros":
        var = F.pad(var, (pad_size, pad_size), "constant", 0)
    elif padding == "replication":
        var = F.pad(var, (pad_size, pad_size), "replicate")

    # Perform convolution
    output = F.conv1d(var, ddx1D, padding='valid')

    # Adjust the output with variable dx
    # Assuming dx has shape [length] or broadcastable to [batch, channels, length]
    output = output / dx.view(1, 1, -1)

    return output

# Example usage:
# inpt = torch.randn(1, 3, 10)  # [batch, channels, length]
# dx_values = torch.linspace(0.1, 1.0, 10)  # example of variable spacing
# output = dx(inpt, dx_values, channel=0, order=1, padding="zeros")

def ddx(inpt, dx, channel, order=1, padding="zeros"):
    "Compute second order numerical derivatives of input tensor with variable spacing"

    # Extract the specified channel
    var = inpt[:, channel : channel + 1, :]
    
    # Ensure dx is a tensor
    if not isinstance(dx, torch.Tensor):
        dx = torch.tensor(dx, device=inpt.device)
    
    # Get the filter
    if order == 1:
        ddx1D = torch.Tensor([1.0, -2.0, 1.0]).to(inpt.device)
    elif order == 3:
        ddx1D = torch.Tensor([1.0 / 90.0, -3.0 / 20.0, 3.0 / 2.0, -49.0 / 18.0, 3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0]).to(inpt.device)

    ddx1D = ddx1D.view(1, 1, -1)  # reshape to [1, 1, filter_length]

    # Apply padding
    pad_size = (ddx1D.shape[2] - 1) // 2
    if padding == "zeros":
        var = F.pad(var, (pad_size, pad_size), "constant", 0)
    elif padding == "replication":
        var = F.pad(var, (pad_size, pad_size), "replicate")

    # Perform convolution
    output = F.conv1d(var, ddx1D, padding='valid')

    # Adjust the output with variable dx
    # Assuming dx has shape [length] or broadcastable to [batch, channels, length]
    output = output / (dx.view(1, 1, -1) ** 2)

    return output

# Example usage:
# inpt = torch.randn(1, 3, 10)  # [batch, channels, length]
# dx_values = torch.linspace(0.1, 1.0, 10)  # example of variable spacing
# output = ddx(inpt, dx_values, channel=0, order=1, padding="zeros")
