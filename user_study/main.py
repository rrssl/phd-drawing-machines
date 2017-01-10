#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User study

@author: Robin Roussel
"""
import json
import random
import matplotlib.pyplot as plt
#from matplotlib.gridspec import GridSpec
#from matplotlib.widgets import Button
import numpy as np

import _context
from controlpane import ControlPane


class App:
    def __init__(self, mecha_type, init_props, target_props, pt_density=2**6):
        self.pt_density = pt_density

        self.mecha = mecha_type(*target_props)
        self.target_crv = self.mecha.get_curve(nb=pt_density)
        self.mecha.reset(*init_props)
        self.crv = self.mecha.get_curve(nb=pt_density)

        self._init_draw()

    def _init_draw(self):
        self.fig = plt.figure(figsize=(10,11))
#        gs = GridSpec(9, 12)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('equal')
        self.ax.margins(.3)
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
#        plt.subplots_adjust(left=.05, wspace=0., hspace=1.)

        self.crv_plot = self.ax.plot(*self.crv, lw=2, alpha=.9)[0]
        self.ax.plot(*self.target_crv, c='k', alpha=.5)

        self.control_pane = self._create_controls()

#        btn_ax = self.fig.add_subplot(gs[-1, 7:9])
#        self.gen_btn = Button(btn_ax, "Generate random\ncombination")
#        self.gen_btn.on_clicked(self.generate_random_params)
#
#        btn_ax = self.fig.add_subplot(gs[-1, 10:])
#        self.sv_btn = Button(btn_ax, "Save combination")
#        self.sv_btn.on_clicked(self.save_params)

#        self.redraw()

    def _create_controls(self):
        data = []
        bounds = []
        ndp = self.mecha.ConstraintSolver.nb_dprops
        ncp = self.mecha.ConstraintSolver.nb_cprops
        for i in range(ndp, ndp+ncp):
            bounds.append(self.mecha.get_prop_bounds(i))
            data.append(
                (i, {'valmin': .5*bounds[-1][0],
                     'valmax': 1.5*bounds[-1][1],
                     'valinit': self.mecha.props[i],
                     'label': ''
                     })
                )
        cp = ControlPane(self.fig, data, self.update, bounds=bounds,
                         show_value=False)
        for s in cp.sliders.values():
            s.drawon = False

        return cp

    def generate_random_params(self, event):
        # Collect static bounds.
        bounds = []
        s = self.control_pane.sliders
        for i in range(len(s)):
            bounds.append((s[i].valmin, s[i].valmax))
        # Find feasible parameters.
        params = [0] * len(bounds)
        feasible = False
        while not feasible:
            for i, (a, b) in enumerate(bounds):
                if type(a) == int and type(b) == int:
                    params[i] = random.randint(a, b)
                else:
                    params[i] = random.random() * (b-a) + a
            feasible = self.mecha.reset(*params)
        # Compute new dynamic bounds.
        for i in range(len(bounds)):
            # Slider id is the same as parameter id.
            self.control_pane.set_bounds(i, self.get_bounds(i))
        # Update view.
        for i, p in enumerate(params):
            self.control_pane.set_val(i, p, incognito=True)
        self.crv = self.mecha.get_curve(nb=self.pt_density)
        self.redraw()
        self.fig.canvas.draw_idle()

    def save_params(self, event):
        save = {
            'type': type(self.mecha).__name__,
            'params': self.mecha.props
            }
        try:
            with open("saved_params.json", "r") as file:
                data = json.load(file)
                data.append(save)
        except FileNotFoundError:
                data = [save]
        with open("saved_params.json", "w") as file:
                json.dump(data, file)
        print('Successfully saved {}'.format(save))

    def get_bounds(self, i):
        a, b = self.mecha.get_prop_bounds(i)
        if (i >= self.mecha.ConstraintSolver.nb_dprops
            and np.isfinite((a,b)).all()):
            # Account for slider imprecision wrt bounds.
            margin = (b - a) / 100.
            a += margin
            b -= margin
        return a, b

    def redraw(self):
        self.crv_plot.set_data(*self.crv)
        self.fig.canvas.draw_idle()

    def run(self):
        plt.ioff()
        plt.show()

    def update(self, pid, val):
        """Update the figure."""
        ndp = self.mecha.ConstraintSolver.nb_dprops
        ncp = self.mecha.ConstraintSolver.nb_cprops
        success = self.mecha.update_prop(pid, val)
        if success:
            for i in range(ndp, ndp+ncp):
                if i != pid:
                    # Slider id is the same as parameter id.
                    self.control_pane.set_bounds(i, self.get_bounds(i))
            self.crv = self.mecha.get_curve(nb=self.pt_density)
            self.redraw()
        else:
            print("Val", val, "with bounds", self.mecha.get_prop_bounds(pid))


def main():
    from mecha import HootNanny
    from _config import task_1

    for subtask_data in task_1:
        app = App(HootNanny, pt_density=2**8, **subtask_data)
        app.run()

if __name__ == "__main__":
    main()
