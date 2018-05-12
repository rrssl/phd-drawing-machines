#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qt5 test

@author: Robin Roussel
"""

import sys

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QApplication, QSizePolicy,
                             QSlider)

import context
from sliders import IntSlider, FloatSlider

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        # Clear the axes every time plot() is called
        self.ax.hold(False)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.redraw_plot()

    def redraw_plot(self):
        raise NotImplementedError


class DynamicPlotCanvas(PlotCanvas):
    """A canvas that gets updated when a new frequency value is given."""

    def __init__(self, *args, **kwargs):
        self.freq = 1
        PlotCanvas.__init__(self, *args, **kwargs)

    def update_freq(self, val):
        self.freq = val
        self.redraw_plot()

    def redraw_plot(self):
        a = np.linspace(0, 20, 100)
        s = np.sin(a / self.freq)
        try:
            self.plot.set_data(a, s)
        except AttributeError:
            self.plot = self.ax.plot(a, s, 'r')
        self.draw_idle()


class App(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):
        sld = IntSlider(Qt.Horizontal, self)
        sld.setTickInterval(1)
        sld.setTickPosition(QSlider.TicksBelow)
        sld.setToolTip("This is a slider")
        m = DynamicPlotCanvas(self, width=5, height=4)
#        m.move(0,0)

        vbox = QVBoxLayout()
        vbox.addWidget(m)
        vbox.addWidget(sld)
        self.setLayout(vbox)

        sld.valueChanged.connect(m.update_freq)
        sld.setValue(5)
        sld.setValidRange(2, 8)
        sld.setRange(1, 10)
        sld.setFocusPolicy(Qt.NoFocus)

        self.setGeometry(10, 10, 640, 400) # left, top, width, height
        self.setWindowTitle('Test')
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
