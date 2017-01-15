#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple PoI selector: click on it, you get the point index / parameter value.

@author: Robin Roussel
"""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import CheckButtons

import _context
import pois


class PoISelector:

    def __init__(self, drawing, id2time):
        """
        Parameters
        ----------
        drawing: 2 x N_points iterable
            The drawing curve.
        id2time: callable
            Used to convert a point ID to the corresponding time parameter.
        """
        self.drw = drawing
        self.id2time = id2time # Method converting IDs to corresp. time params.
        self.opt_labels = ('Show curvature maxima', 'Show intersections')
        self._init_data()
        self._init_draw()
        self._init_ctrl()

    def _init_data(self):
        self.pois = (
            pois.find_krv_max(self.drw),
            pois.find_isect(self.drw)
            )
        self.pois_slices = ( # Column(s) where to find the PoI index(es).
            slice(2, 3),
            slice(2, 4)
            )

    def _init_draw(self):
        self.fig = plt.figure(figsize=(10, 10))
        gs = GridSpec(8, 4)
        self.ax = self.fig.add_subplot(gs[:-1, :])
        self.ax.set_aspect('equal')
        self.ax.margins(.1)
        plt.subplots_adjust(left=0, right=1, bottom=.05, top=.95)
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])

        self.drw_plot = self.ax.plot(*self.drw, lw=1, alpha=.8)[0]

        self.opt_btns = CheckButtons(
            self.fig.add_subplot(gs[-1, 1:-1]), self.opt_labels, (True, True))
        colors = 'r', 'g'
        self.pois_plots = [
            # Scatter is better than plot here because in the latter, lines
            # would fire a PickEvent as well (even if they have 0 width).
            self.ax.scatter(*pois[:, :2].T, s=50, c=c, edgecolor='w', zorder=3,
                            picker=True)
            for pois, c in zip(self.pois, colors)
            ]

        self.redraw()

    def _init_ctrl(self):
        self.poi_type = None
        self.opt_btns.on_clicked(self.on_btn_press)

        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
#        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_pick(self, event):
        # If a PoI is clicked, give its ID
        if event.artist.get_visible():
            poi_type = self.pois_plots.index(event.artist)
            poi_id = self.pois[poi_type][event.ind, self.pois_slices[poi_type]]
            print("PoI(s) selected")
            print('ID\n', poi_id.astype(int))
            print('time\n', self.id2time(poi_id))
            print('------------')

    def on_btn_press(self, label):
        plot = self.pois_plots[self.opt_labels.index(label)]
        plot.set_visible(not plot.get_visible())
        self.redraw()

    def redraw(self):
        self.fig.canvas.draw_idle()

    def run(self):
        plt.ioff()
        plt.show()

def main():
    import mecha
#    m = mecha.HootNanny(
#        *[15, 10, 2, 1.3433724430459493, 1.9058189313461327, 1.98, 18.500079276993844, 17.13017282384655])
#    drawing = m.get_curve(2**7)
    m = mecha.Thing(
        *[0.09191176470588247, 0.1663602941176472, 0.08226102941176472, 0.020220588235294157, 0.38419117647058854])
    drawing = m.get_curve(2**8)
    app = PoISelector(drawing, m.id2time)
    app.run()

if __name__ == "__main__":
    main()
