# -*- coding: utf-8 -*-
"""
Forward simulation of a Spirograph.

@author: Robin Roussel
"""
import context
from _base import ForwardController
from mecha import BaseSpirograph as mecha_type
from _config import spiro_data as data

def main():
    app = ForwardController(mecha_type, data)
    app.run()

if __name__ == "__main__":
    main()
