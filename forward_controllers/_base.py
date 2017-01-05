# -*- coding: utf-8 -*-
"""
Base class for the forward controller.

@author: Robin Roussel
"""
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
import numpy as np

import context
from mechaplot import mechaplot_factory
from controlpane import ControlPane


class ForwardController:
    def __init__(self, mecha_type, param_data, pt_density=2**6):
        self.param_data = param_data
        self.pt_density = pt_density

        self.mecha = mecha_type(
            *[d.get('valinit') for _, d in param_data])
        self.crv = self.mecha.get_curve(nb=self.pt_density)

        self._init_draw(param_data)

    def _init_draw(self, param_data):
        self.fig = plt.figure(figsize=(16,8))
        gs = GridSpec(9, 6)
        self.ax = self.fig.add_subplot(gs[:, :3])
        self.ax.set_aspect('equal')
#        self.ax.get_xaxis().set_ticks([])
#        self.ax.get_yaxis().set_ticks([])
        plt.subplots_adjust(left=.05, wspace=0., hspace=1.)

        self.crv_plot = self.ax.plot([], [], lw=1, alpha=.8)[0]
        # Since the paper may rotate with the turntable, we pass the drawing.
        self.mecha_plot = mechaplot_factory(self.mecha, self.ax, self.crv_plot)

        bounds = [self.get_bounds(i) for i in range(len(self.mecha.props))]
        self.control_pane = ControlPane(self.fig, param_data, self.update,
                                        subplot_spec=gs[:-2, 4:], bounds=bounds)

        btn_ax = self.fig.add_subplot(gs[-1, 4:])
        self.gen_btn = Button(btn_ax, "Generate random combination")
        self.gen_btn.on_clicked(self.generate_random_params)

        self.redraw()

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
        self.mecha_plot.redraw()

    def run(self):
        plt.ioff()
        plt.show()

    def update(self, pid, val):
        """Update the figure."""
        success = self.mecha.update_prop(pid, val)
        if success:
            for i in range(len(self.mecha.props)):
                if i != pid:
                    # Slider id is the same as parameter id.
                    self.control_pane.set_bounds(i, self.get_bounds(i))
            self.crv = self.mecha.get_curve(nb=self.pt_density)
            self.redraw()
        else:
            print("Val", val, "with bounds", self.mecha.get_prop_bounds(pid))
