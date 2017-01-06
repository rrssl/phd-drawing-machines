#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
App allowing direct exploration of the feasible design space of a machine.

@author: Robin Roussel
"""
import _context
from _base import ForwardController

#import warnings
#warnings.filterwarnings("error")

pt_density = 2**6
if 0:
    from mecha import BaseSpirograph as mecha_type
    from _config import spiro_data as data
elif 0:
    from mecha import EllipticSpirograph as mecha_type
    from _config import ellip_data as data
elif 0:
    from mecha import SingleGearFixedFulcrumCDM as mecha_type
    from _config import cdm_data as data
elif 1:
    from mecha import HootNanny as mecha_type
    from _config import hoot_data as data
    pt_density = 2**10
elif 0:
    from mecha import Kicker as mecha_type
    from _config import kick_data as data


def main():
    app = ForwardController(mecha_type, data, pt_density)
    app.run()

if __name__ == "__main__":
    main()
