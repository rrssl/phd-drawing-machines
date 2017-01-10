#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple explorer of saved drawings.

@author: Robin Roussel
"""
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['keymap.back'].remove('left')
mpl.rcParams['keymap.forward'].remove('right')

import _context
import mecha


class DrawingViewer:
    def __init__(self, filename, id_):
        with open(filename, "r") as file:
            self.db = json.load(file)
        print("Number of drawings: {}".format(len(self.db)))
        self.id_ = id_
        self.drw = self.get_drawing(self.db[id_])
        self._init_draw()
        self._init_ctrl()

        # Output all
#        for i, drw in enumerate(self.db):
#            self.drw = self.get_drawing(drw)
#            title = 'drawing_{}'.format(i)
#            self.ax.set_title(title + '\n')
#            self.redraw()
#            plt.savefig(title)

    def _init_draw(self):
        self.fig = plt.figure(figsize=(8,8))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('equal')
        self.ax.margins(.1)
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.fig.subplots_adjust(left=.01, right=.99, bottom=0.01, top=.99)

        self.drw_plot = self.ax.plot([], [], lw=1, alpha=.8)[0]

        self.redraw()

    def _init_ctrl(self):
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def get_drawing(self, data):
        print(data)
        cls = getattr(mecha, data['type'])
        mch = cls(*data['params'])
        return mch.get_curve(2**10)

    def on_key_press(self, event):
        redraw = False
        if event.key == 'left':
            self.id_ = (self.id_ - 1) % len(self.db)
            redraw = True
        if event.key == 'right':
            self.id_ = (self.id_ + 1) % len(self.db)
            redraw = True
        if redraw:
            self.drw = self.get_drawing(self.db[self.id_])
            self.redraw()

    def redraw(self):
        self.drw_plot.set_data(*self.drw)
        self.ax.relim()
        self.ax.autoscale()
        self.fig.canvas.draw_idle()

    def run(self):
        plt.ioff()
        plt.show()

def main():
    app = DrawingViewer("saved_params.json", 0)
    app.run()

if __name__ == "__main__":
    main()
