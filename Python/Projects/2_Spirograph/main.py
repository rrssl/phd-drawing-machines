# -*- coding: utf-8 -*-
"""
Project 2: Spirograph

Author: Robin Roussel
"""
import matplotlib.pyplot as plt
from spiroplots import SpiroPlot, SpiroGridPlot

if __name__ == "__main__":
    plt.ioff()
#    sp = SpiroPlot(show_spiro=True)
#    sp.show()

    sgp = SpiroGridPlot()
    sgp.show()
