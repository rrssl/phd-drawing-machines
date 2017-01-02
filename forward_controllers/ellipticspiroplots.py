# -*- coding: utf-8 -*-
"""
Simulation and control of the elliptic spirograph.

@author: Robin Roussel
"""
import context
from _base import ForwardController
from mecha import EllipticSpirograph as mecha_type
from _config import ellip_data as data


def main():
    app = ForwardController(mecha_type, data)
    app.run()

if __name__ == "__main__":
    main()
