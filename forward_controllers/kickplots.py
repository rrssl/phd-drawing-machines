#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forward simulation of a ball-kicking toy.

@author: Robin Roussel
"""
import context
from _base import ForwardController
from mecha import Kicker as mecha_type
from _config import kick_data as data

def main():
    app = ForwardController(mecha_type, data)
    app.run()

if __name__ == "__main__":
    main()
