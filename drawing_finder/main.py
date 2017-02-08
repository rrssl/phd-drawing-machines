#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo for sketch-based curve retrieval.

@author: Robin Roussel
"""
#import warnings
#warnings.filterwarnings("error")
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QApplication, QSizePolicy,
                             QSlider, QPushButton, QLabel, QFrame,
                             QVBoxLayout, QHBoxLayout)

import _context
import mecha
TYPES = mecha.EllipticSpirograph, # mecha.SingleGearFixedFulcrumCDM
import curvedistances as cdist
from sketcher import Sketcher

DEBUG = True
if DEBUG:
    from curveplotlib import distshow

class DrawingFinder:

    def __init__(self):
        ## Sketcher
        self.crv_bnds = [None, None]
        self.sym_order = 1
        self.strokes = [] # List of N*2 lists of points
        self.undone_strokes = []
        # Mechanism retrieval
        self.pts_per_dim = 4
        self.samples = self.get_global_sampling()
        self.distance = cdist.DistanceField().get_dist
        self.search_res = [] # list of  {'type', 'props', 'curve'} dicts
        # Mechanism
        self.mecha = None
        self.nb_crv_pts = 2**6
        self.crv = None

    def get_global_sampling(self):
        """Sample feasible parameters across all mechanisms."""
        samples = {}
        for t in TYPES:
            size = [self.pts_per_dim]*t.ConstraintSolver.nb_cprops
            samples[t] = np.array(list(
                t.ConstraintSolver.sample_feasible_domain(grid_resol=size)))
        return samples

    def search_mecha(self, nb):
        """Retrieve the closest drawings and their associated mechanisms."""
        if not len(self.strokes):
            return
        # TODO FIXME 'ValueError: bad axis1 argument to swapaxes'
        # with more than 1 symmetrized stroke
        sketch = np.array(self.strokes).swapaxes(1, 2)

        self.search_res.clear()
        ranges = [0]
        # Convert types and samples to lists to keep the order.
        types = list(self.samples.keys())
        samples = list(self.samples.values())
        # Pre-filter the samples.
        if self.sym_order > 1:
            samples = [s[s[:, 0] == self.sym_order] for s in samples]
        # Compute distances.
        distances = []
        for type_, type_samples in zip(types, samples):
            ranges.append(ranges[-1] + type_samples.shape[1])
            mecha = type_(*type_samples[0])
            for sample in type_samples:
                mecha.reset(*sample)
                crv = mecha.get_curve(self.nb_crv_pts)
                distances.append(max(self.distance(crv, sketch),
                                     self.distance(sketch, crv)))
        distances = np.array(distances)
        best = distances.argpartition(nb)[:nb]
        # Sort the best matching curves.
        best = best[distances[best].argsort()]
        print(distances[best])
        # Build the result.
        ranges = np.array(ranges)
        for id_ in best:
            # Find index in ranges with a small trick: argmax gives id of the
            # first max value, here True.
            # TODO FIXME
#            typeid = np.argmax(ranges > id_) - 1
#            print(id_, id_-ranges[typeid])
#            type_ = types[typeid]
#            mecha = type_(*samples[typeid][id_-ranges[typeid]])
            type_ = types[0]
            mecha = type_(*samples[0][id_])
            self.search_res.append({
                'type': type_,
                'props': mecha.props.copy(),
                'curve': mecha.get_curve(self.nb_crv_pts)
                })


def HLine():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


class Window(QWidget):

    def __init__(self):
        super().__init__()

        self.finder = DrawingFinder()
        self.initUI()

        self.show()

    def initUI(self):
        # Define matplotlib context for the sketcher.
        fig = Figure(figsize=(6, 6), dpi=100)
        can = FigureCanvas(fig)
        can.setParent(self)
        can.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        can.updateGeometry()
        # Add the sketcher.
        fig.subplots_adjust(left=0., right=1., bottom=0., top=1.,
                            wspace=0., hspace=0.)
        ax = fig.add_subplot(111)
        self.sketcher = Sketcher(ax, self.finder)
        # Add buttons.
        bnd_bt = QPushButton("Set bounds", self)
        undo_bt = QPushButton("Undo stroke", self)
        redo_bt = QPushButton("Redo stroke", self)
        search_bt = QPushButton("Search drawing", self)
        # Add slider.
        sym_sld = QSlider(Qt.Horizontal, self)
        sym_sld.setTickInterval(1)
        sym_sld.setTickPosition(QSlider.TicksBelow)
        sym_sld.setToolTip("Change the order of rotational symmetry")
        sym_sld.setValue(1)
        sym_sld.setRange(0, 10)
        sym_sld.setFocusPolicy(Qt.NoFocus)
        # Add text box.
        sym_txt = QLabel("Symmetry order", self)
        sym_txt.setAlignment(Qt.AlignCenter)
        # Connect callbacks.
        bnd_bt.clicked.connect(self.sketcher.set_sketch_bounds)
        undo_bt.clicked.connect(self.sketcher.undo_stroke)
        redo_bt.clicked.connect(self.sketcher.redo_stroke)
        search_bt.clicked.connect(self.search_mecha)
        sym_sld.valueChanged.connect(self.sketcher.set_symmetry)
        # Arrange widgets.
        top = QHBoxLayout()
        top.addWidget(can)
        panel = QVBoxLayout()
        top.addLayout(panel)
        panel.addSpacing(10)
        panel.addWidget(bnd_bt)
        panel.addSpacing(10)
        panel.addWidget(sym_txt)
        panel.addSpacing(-10)
        panel.addWidget(sym_sld)
        panel.addSpacing(10)
        undoredo = QHBoxLayout()
        panel.addLayout(undoredo)
        undoredo.addWidget(undo_bt)
        undoredo.addWidget(redo_bt)
        panel.addSpacing(30)
        panel.addWidget(HLine())
        panel.addWidget(search_bt)
        panel.addWidget(HLine())
        panel.addStretch()
        self.setLayout(top)
        # Finalize window
        self.setGeometry(10, 10, 800, 600) # left, top, width, height
        self.setWindowTitle("Drawing finder")

    def search_mecha(self, event):
        if not len(self.finder.strokes):
            print("There is no query sketch.")
        else:
            print("Search for the best matching mechanism.")
            self.finder.search_mecha(6)
            if DEBUG:
                # TODO FIXME distance field is sometimes broken
                sketch = np.array(self.finder.strokes).swapaxes(1, 2)
                fig = plt.figure(figsize=(6,12))
                ax1 = fig.add_subplot(211)
                ax1.set_aspect('equal')
                for stroke in sketch:
                    ax1.plot(*stroke, c='b', lw=2)
                ax2 = fig.add_subplot(212)
                distshow(ax2, sketch)
                fig.tight_layout()
                fig.show()

                fig = plt.figure(figsize=(12,6))
                for i, sr in enumerate(self.finder.search_res):
                    ax = fig.add_subplot(2, 3, i+1)
                    distshow(ax, sketch, sr['curve'])
                fig.tight_layout()
                fig.show()


def main():
    app = QApplication(sys.argv)
    w = Window()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
