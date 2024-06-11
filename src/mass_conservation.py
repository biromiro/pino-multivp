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

from sympy import Symbol, Function, diff

from modulus.sym.eq.pde import PDE
from modulus.sym.node import Node


class MassConservation(PDE):
    """
    Mass conservation system of equations
    """
    name = "Mass_Conservation"

    def __init__(self, n='n', v='v', T='T'):
        # set params
        self.n = n
        self.v = v
        self.T = T
              
        # coordinates
        L = Symbol("L")
        B = Function('B')(L)

        # make input variables
        input_variables = {"L": L, "B": B}

        # density
        assert type(n) == str, "n needs to be string"
        n = Function(n)(*input_variables)
        
        # velocity
        assert type(v) == str, "v needs to be string"
        v = Function(v)(*input_variables)
        
        # Temperature
        assert type(T) == str, "T needs to be string"
        T = Function(T)(*input_variables)
        

        # set equations
        self.equations = {}
        self.equations["mass_term"] = (
            diff((n*v)/B, L, 1)
        )
        