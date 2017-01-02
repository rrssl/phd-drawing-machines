#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation and control of the "Hoot-Nanny Magic Designer".

@author: Robin Roussel
"""
import context
from _base import ForwardController
from mecha import HootNanny as mecha_type
from _config import hoot_data as data

def main():
    app = ForwardController(mecha_type, data, pt_density=2**10)
    app.run()

if __name__ == "__main__":
    main()
