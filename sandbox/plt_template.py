# -*- coding: utf-8 -*-
"""
Pyplot app boilerplate.

@author: Robin Roussel
"""
import matplotlib.pyplot as plt


class MyApp:
    """My app."""
    
    def __init__(self):
        self.init_draw()
    
    def init_draw(self):
        """Initialize the canvas."""
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
    
    def redraw(self):
        """Redraw."""
        pass

def main():
    """Entry point."""
    plt.ioff()

    MyApp()

    plt.show()

main()
