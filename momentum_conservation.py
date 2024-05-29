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

"""Diffusion equation
"""

from sympy import Symbol, Function, Number, diff
from sympy.stats import Variance


from modulus.sym.eq.pde import PDE
from modulus.sym.node import Node


class MomentumConservation(PDE):
    """
    Momentum conservation system of equations
    """

    name = "Momentum_Conservation"

    def __init__(self, n='n', v='v', T='T'):
        # set params
        self.n = n
        self.v = v
        self.T = T
        self.G = 6.6743e-11
        # self.sun_mass = 1.989e+30
        DX_UN  =   6.96000e+10  # length unit factor (to cm)
        VX_UN  =   1e+05 # speed unit factor (to cm/s)
        TE_UN  =   1.00000e+06  # temperature unit factor (to K)
        NE_UN  =   1.00000e+17  # density unit factor (to 1/cm^3)
        DT_UN  =       5413.93  # time unit factor (to s)

        NU_UN = (DX_UN**2)/DT_UN   # viscosity coeff init factor (to cm^2/s)

        NU_VISC = .1 * NU_UN    # the actual viscosity coeff
        self.nu_visc = NU_VISC
        
        # coordinates
        R, L, B, a_a0, cos_alpha = Symbol("R"), Symbol("L"), Symbol("B"), Symbol("a_a0"), Symbol("cos_alpha")

        # make input variables
        input_variables = {"R": R, "L": L, "B": B, "a_a0": a_a0, "cos_alpha": cos_alpha}

        # density
        assert type(n) == str, "n needs to be string"
        n = Function(n)(*input_variables)
        
        # velocity
        assert type(v) == str, "v needs to be string"
        v = Function(v)(*input_variables)
        
        # Temperature
        assert type(T) == str, "T needs to be string"
        T = Function(T)(*input_variables)
        
        G = Number(self.G)
        nu = Number(self.nu_visc)

        # set equations
        self.equations = {}
        self.equations["mom_term"] = (
            diff(n * T, L, 1) * n
            + G * cos_alpha / (R ** 2)
            - v * diff(v, L, 1)
            - nu * (diff(v, L, 2) + a_a0 * diff(v, L, 1))
        )