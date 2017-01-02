#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation and display of the Cycloid Drawing Machine.

@author: Robin Roussel
"""
import context
from _base import ForwardController
from mecha import SingleGearFixedFulcrumCDM as mecha_type
from _config import cdm_data as data

def main():
    app = ForwardController(mecha_type, data)
    app.run()

if __name__ == "__main__":
    main()
