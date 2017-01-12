#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User study

@author: Robin Roussel
"""
import json
import random
import time
import matplotlib.pyplot as plt
#from matplotlib.gridspec import GridSpec
#from matplotlib.widgets import Button
import numpy as np

import _context
from controlpane import ControlPane
from invarspace import InvariantSpaceFinder

# TODO : replace is-a with has-a
class TaskManager(InvariantSpaceFinder):
    def __init__(self, cand_name, **params):
        self.cand_name = cand_name
        self.taskid = params.pop('taskid')
        self.subtask_params = params.copy()
        # Initialize the invariant explorer.
        self.get_corresp = params.pop('get_corresp')
        self.get_features = params.pop('get_features')
        super().__init__(props=params.pop('target_props'),
                         nb_crv_pts=params.pop('pt_density'), **params)
        # Generate random initial point.
        self.randomize_cont_props()
        self.subtask_params['init_props'] = self.mecha.props.copy()
        # Define and randomize subtasks.
        self.subtasks = [ExploreBaseSpace, ExploreInvarSpace]
        random.shuffle(self.subtasks)
        # Initialize user data.
        get_subtask_data = lambda: { # use lambda to avoid shallow copy
            'tot_time': 0.,
            'cont_props': [] # Continuous props at each step
            }
        self.data = {
            'taskid': self.taskid,
            'init_props': tuple(self.mecha.props),
            'subtask_data': {
                # The values of this dict are passed to each sub_task
                ExploreBaseSpace.__name__: get_subtask_data(),
                ExploreInvarSpace.__name__: get_subtask_data()
                }
            }

    def randomize_cont_props(self):
        # Collect bounds.
        ndp = len(self.disc_prop)
        ncp = len(self.cont_prop)
        dp = list(self.disc_prop)
        bounds = [self.mecha.get_prop_bounds(i)
                  for i in range(ndp, ndp+ncp)]
        # Find feasible parameters.
        feasible = False
        while not feasible:
            cp = [random.random() * (b-a) + a for a, b in bounds]
            feasible = self.mecha.reset(*dp+cp)
        cp = list(self.project_cont_prop_vect())
        self.mecha.reset(*dp+cp)

    def save_data(self):
        filename = self.cand_name + "_task_" + str(self.taskid) + ".json"
        with open(filename, "w") as file:
                json.dump(self.data, file)
        print("Successfully saved {}".format(filename))

    def run(self):
        print("Starting subtasks of task {}".format(self.taskid))
        for subtask in self.subtasks:
            data = self.data['subtask_data'][subtask.__name__]
            app = subtask(data, **self.subtask_params)
            app.run()
        self.save_data()


class Subtask:
    def __init__(self, user_data, mecha_type, init_props, target_props,
                 pt_density=2**6,  **kwargs):
        self.pt_density = pt_density
        self.user_data = user_data

        # Create mecha, new_curve, and target_curve here

        self._init_draw()
        self._init_ctrl()

    def _init_draw(self):
        self.fig = plt.figure(figsize=(10,11))
#        gs = GridSpec(9, 12)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('equal')
        self.ax.margins(.3)
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
#        plt.subplots_adjust(left=.05, wspace=0., hspace=1.)

        self.crv_plot = self.ax.plot(*self.new_crv, lw=2, alpha=.9)[0]
        self.ax.plot(*self.target_crv, c='k', alpha=.5)

#        self.redraw()

    def _init_ctrl(self):
#        btn_ax = self.fig.add_subplot(gs[-1, 7:9])
#        self.gen_btn = Button(btn_ax, "Generate random\ncombination")
#        self.gen_btn.on_clicked(self.generate_random_params)
#
#        btn_ax = self.fig.add_subplot(gs[-1, 10:])
#        self.sv_btn = Button(btn_ax, "Save combination")
#        self.sv_btn.on_clicked(self.save_params)
        self.control_pane = self._create_controls()
        self.slider_active = -1
        self.fig.canvas.mpl_connect('button_release_event',
                                    self.on_button_release)

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
        self.crv_plot.set_data(*self.new_crv)
        self.fig.canvas.draw_idle()

    def run(self):
        plt.ioff()
        start = time.time()
        plt.show()
        stop = time.time()
        self.user_data['tot_time'] = stop - start
        print('Done')

    def save_step(self, type_):
        """type_ = 's' for slider or 'p' for projection."""
        self.user_data['cont_props'].append(
            (tuple(self.mecha.props[self.mecha.constraint_solver.nb_dprops:]),
             type_)
            )


class ExploreBaseSpace(Subtask):
    def __init__(self, user_data, mecha_type, init_props, target_props,
                 pt_density=2**6,  **kwargs):
        self.pt_density = pt_density
        self.user_data = user_data

        self.mecha = mecha_type(*target_props)
        self.target_crv = self.mecha.get_curve(nb=pt_density)
        self.mecha.reset(*init_props)
        self.new_crv = self.mecha.get_curve(nb=pt_density)

        self._init_draw()
        self._init_ctrl()

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

    def on_button_release(self, event):
        """Callback function for mouse button release."""
        pid = self.slider_active
        if pid >= 0:
            self.slider_active = -1

            # Reset bounds
            ndp = self.mecha.ConstraintSolver.nb_dprops
            ncp = self.mecha.ConstraintSolver.nb_cprops
            for i in range(ndp, ndp+ncp):
                if i != pid:
                    # Slider id is the same as parameter id.
                    self.control_pane.set_bounds(i, self.get_bounds(i))

            self.save_step('s')
            self.redraw()

    def update(self, pid, val):
        """Update the figure."""
        success = self.mecha.update_prop(pid, val)
        if success:
            self.slider_active = pid

#            ndp = self.mecha.ConstraintSolver.nb_dprops
#            ncp = self.mecha.ConstraintSolver.nb_cprops
#            for i in range(ndp, ndp+ncp):
#                if i != pid:
#                    # Slider id is the same as parameter id.
#                    self.control_pane.set_bounds(i, self.get_bounds(i))
            self.new_crv = self.mecha.get_curve(nb=self.pt_density)
            self.redraw()
        else:
            print("Val", val, "with bounds", self.mecha.get_prop_bounds(pid))


class ExploreInvarSpace(InvariantSpaceFinder, Subtask):
    def __init__(self, user_data, mecha_type, init_props, target_props,
                 pt_density=2**6,  **kwargs):
        self.pt_density = pt_density
        self.user_data = user_data

        self.mecha = mecha_type(*target_props)
        self.target_crv = self.mecha.get_curve(nb=pt_density)

        self.get_corresp = kwargs.pop('get_corresp')
        self.get_features = kwargs.pop('get_features')
        super().__init__(
            mecha_type, init_props, nb_crv_pts=pt_density, **kwargs)
        self.new_crv = self.mecha.get_curve(nb=pt_density)

        self._init_draw()
        self._init_ctrl()

    def _create_controls(self):
        """Create the controls to explore the invariant space."""
        data = [
            (i, {'valmin': -2.,
                 'valmax': 2.,
                 'valinit': self.cont_prop_invar_space[i],
                 'label': ''
                 })
            for i in range(self.ndim_invar_space)
            ]

        cp = ControlPane(self.fig, data, self.update,
                         bounds=self.bnds_invar_space, show_value=False)
        for s in cp.sliders.values():
            s.drawon = False

        return cp

    def _set_sliders_active(self, state):
        for s in self.control_pane.sliders.values():
            s.set_active(state)

    def on_button_release(self, event):
        """Callback function for mouse button release."""
        pid = self.slider_active
        if pid >= 0:
            self.save_step('s')

            self.slider_active = -1
            # Update title
            self.ax.set_title("Please wait\n")
            plt.pause(.001)
            # Project position and recompute invariant approximation.
            cont_prop = self.project_cont_prop_vect()
            self.set_cont_prop(cont_prop)
            self.compute_invar_space()
            # Update sliders.
            for i, val in enumerate(self.cont_prop_invar_space):
                self.control_pane.set_val(i, val, incognito=True)
                self.control_pane.set_bounds(i, self.bnds_invar_space[i])
            # Flush sliders events that happened during computation.
            self._set_sliders_active(False)
            plt.pause(.1)
            self._set_sliders_active(True)
            # Update title
            self.ax.set_title("")

            self.save_step('p')

            self.redraw()

    def update(self, pid, val):
        """Update the figure."""
        self.slider_active = pid

        self.cont_prop_invar_space[pid] = val
        cont_prop = self.phi(self.cont_prop_invar_space).ravel()
        self.set_cont_prop(cont_prop)
        # Update slider bounds.
#        for i in range(self.ndim_invar_space):
#            if i != pid:
#                bnds = self.get_bounds_invar_space(i)
#                self.control_pane.set_bounds(i, bnds)

        self.redraw()


def main():
    from _config import tasks, cand_name

    for task_data in tasks:
        tm = TaskManager(cand_name, **task_data)
        tm.run()

if __name__ == "__main__":
    main()
