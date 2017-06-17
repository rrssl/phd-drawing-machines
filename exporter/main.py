#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export script.

@author: Robin Roussel
"""
import datetime

from _hoot import export_hoot
from _thing import export_thing
from _spiro import export_spiro, export_ellip_spiro


EXPORT_SPIRO = False
EXPORT_ELLIP_SPIRO = False
EXPORT_HOOT = False
EXPORT_THING = True


def main():
#    base = "svg/{}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M") + "/"
    base = "svg/{}/" + datetime.datetime.now().strftime("%Y%m%d") + "/"
    name = "{}.svg"


    if EXPORT_SPIRO:
        export_spiro(base, name)

    if EXPORT_ELLIP_SPIRO:
        export_ellip_spiro(base, name)

    if EXPORT_HOOT:
#        export_hoot((10, 4, 2, .9, 2.5, 1.5, 10, 9.95),
#                    1., base, name)
        export_hoot((15, 14, 3, 1.3316335903703507, 3.7699182132015086, 2.0515806228896993, 32.6563894503349, 16.11314752952731),
                    .5, base, name)
#        export_hoot((15, 15, 11, 1.4828358173458094, 6.662438560329804, 4.188996353258297, 34.33010940225151, 30.81963260425178),
#                    .5, base, name)

    if EXPORT_THING:
        export_thing((0.09191176470588247, 0.1663602941176472, 0.08226102941176472, 0.020220588235294157, 0.38419117647058854),
                     8, base, name)


if __name__ == "__main__":
    main()
