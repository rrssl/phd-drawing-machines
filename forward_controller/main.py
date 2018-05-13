"""
App allowing direct exploration of the feasible design space of a machine.


Parameters
----------
mid : int
  Mechanism.
pt_density : int, optional
  Density of the curve.

"""
import sys

import _context  # noqa
from _base import ForwardController


def main():
    if len(sys.argv) < 2:
        print("Please choose a mechanism")
        return
    mid = int(sys.argv[1])
    if len(sys.argv) < 3:
        pt_density = 2**6
    else:
        pt_density = int(sys.argv[2])

    if mid == 0:
        from mecha import BaseSpirograph as mecha_type
        from _config import spiro_data as data
    elif mid == 1:
        from mecha import EllipticSpirograph as mecha_type
        from _config import ellip_data as data
    elif mid == 2:
        from mecha import SingleGearFixedFulcrumCDM as mecha_type
        from _config import cdm_data as data
    elif mid == 3:
        from mecha import HootNanny as mecha_type
        from _config import hoot_data as data
    elif mid == 4:
        from mecha import Kicker as mecha_type
        from _config import kick_data as data
    elif mid == 5:
        from mecha import Thing as mecha_type
        from _config import thing_data as data
    else:
        print("Invalid mechanism")
        return

    app = ForwardController(mecha_type, data, pt_density)
    app.run()


if __name__ == "__main__":
    main()
