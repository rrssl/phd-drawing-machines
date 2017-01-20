# -*- coding: utf-8 -*-
"""
Base classes for smart editing demos.

@author: Robin Roussel
"""
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import numpy.polynomial.polynomial as npol
import scipy.interpolate as interp
import scipy.optimize as opt

import context
from controlpane import ControlPane
from invarspace import InvariantSpaceFinder
from mecha import EllipticSpirograph
from mechaplot import mechaplot_factory


SHOW_OPTIONS = True


def get_dist(ref, cand):
    """Get the L2 distance from each cand point to ref point."""
    ref = np.asarray(ref)
    cand = np.asarray(cand)
    return np.linalg.norm(cand - ref.reshape((-1, 1)), axis=0)


def make_frames_from_slider(filename, slider, nb_frames=15):
    """Save a batch of figure screenshots from successive slider values."""
    if slider.slidermin is not None:
        min_ = slider.slidermin.val
    else:
        min_ = slider.valmin
    if slider.slidermax is not None:
        min_ = slider.slidermax.val
    else:
        max_ = slider.valmax
    rng = np.linspace(min_, max_, nb_frames)
    for i, val in enumerate(rng):
        slider.set_val(val)
        plt.savefig("{}_{0:02d}".format(filename, i))


def _get_inwards_normal(crv, id_):
    tangent = crv[:, id_+1] - crv[:, id_-1]
    normal = np.array([-tangent[1], tangent[0]])
    normal *= np.sign(normal.dot(crv[:, id_+1] - crv[:, id_]))
    return normal / np.linalg.norm(normal)


class DemoOptions:
    """Options for the demos."""
    state = {
        'show_mecha': False
        }

    def __init__(self, ax, demo):
        self.ax = ax
        self.demo = demo

    def hide_ref(self, event):
        for attr in self.demo.__dict__.keys():
            if attr.startswith('ref') and attr.endswith('plt'):
                ref_plt = getattr(self.demo, attr)
                ref_plt.set_visible(not ref_plt.get_visible())

        self.ax.figure.canvas.draw_idle()

    def show_mecha(self, event):
        self.state['show_mecha'] = not self.state['show_mecha']

        try:
            mecha_plt = self.demo.mecha_plt
        except AttributeError:
            mecha_plt = mechaplot_factory(self.demo.mecha, self.ax)
            self.demo.mecha_plt = mecha_plt
        mecha_plt.set_visible(self.state['show_mecha'])
        if self.state['show_mecha']:
            mecha_plt.redraw()
        else:
            self.ax.autoscale()

        self.ax.figure.canvas.draw_idle()

    def show_before_after(self, event):
        if self.demo.new_crv is not None:
            fig = plt.figure(2, figsize=(16,8))

            ax1 = fig.add_subplot(121)
            ax1.set_aspect('equal')
            ax1.margins(.1)
            ax1.plot(*self.demo.ref_crv, c='b', lw=2, alpha=.9)
            ax1.scatter(*self.demo.ref_poi, s=100, c='b', alpha=.9, marker='o',
                        edgecolor='w', zorder=3)

            ax2 = fig.add_subplot(122)
            ax2.set_aspect('equal')
            ax2.margins(.1)
            ax2.plot(*self.demo.new_crv, c='r', lw=2, alpha=.9)
            ax2.scatter(*self.demo.new_poi, s=100, c='r', alpha=.9,
                        marker='o', edgecolor='w', zorder=3)

            plt.show()


class InvarDemo(InvariantSpaceFinder):
    """Find the invariant subspace.

    Attributes
    ----------
    disc_prop
    cont_prop
    nb_crv_pts
    mecha

    pts_per_dim
    keep_ratio

    ref_crv
    ref_par
    ref_poi

    new_crv
    new_poi

    phi
    bnds_invar_space

    labels
    """

    ### VIEW

    def init_draw(self):
        """Initialize canvas."""
        self.fig = plt.figure(figsize=(16*1.1, 9*1.1))
        gs = GridSpec(14, 2)
        self.ax = [
            self.fig.add_subplot(gs[:-4, 0]),
            self.fig.add_subplot(gs[:-4, 1]),
            ]
        self.fig.subplots_adjust(left=0.1, right=0.96, bottom=.05, wspace=.3)
        self.cmap = plt.cm.YlGnBu_r

        self.ref_crv_plt = None
        self.ref_poi_plt = None
        self.new_crv_plt = None
        self.new_poi_plt = None
        self.draw_curve_space(self.ax[0])

        self.control = self.create_controls(gs[-3:-1, 0])
        self.ref_control = self.create_ref_controls(gs[-3:-1, 1])
        if SHOW_OPTIONS:
            self.options, self.buttons = self.create_options(self.ax[0])

        for slider in self.control.sliders.values():
            slider.drawon = False

    def draw_curve_space(self, frame):
        """Draw the curve."""
        frame.set_aspect('equal')
        frame.margins(0.3)
        frame.set_xlabel('$x$')
        frame.set_ylabel('$y$')
        frame.set_title("Curve space (visible in the UI).\n")
        # Draw the reference curve...
        self.ref_crv_plt = frame.plot(*self.ref_crv, c='k', alpha=.5,
                                      label="Ref. curve")[0]
        # ... and its point of interest.
        self.ref_poi_plt = frame.scatter(*self.ref_poi, s=100, c='k', alpha=.5,
                                         marker='o', edgecolor='w', zorder=3,
                                         label="Ref. pt(s) of interest")
        # Draw the new curve (empty for now)...
        self.new_crv_plt = frame.plot([], [], 'b-', alpha=.9, lw=2,
                                      label="New curve")[0]
        # ... and its point of interest.
        self.new_poi_plt = frame.scatter([], [], s=100, c='b', marker='o',
                                         edgecolor='w', zorder=3,
                                         label="New pt(s) of interest")
        # Draw the legend.
        frame.legend(loc='upper left', scatterpoints=1, ncol=2,
                     fontsize='medium')

    def create_controls(self, subplot_spec):
        """Create the controls to explore the invariant space."""
        raise NotImplementedError

    def create_ref_controls(self, subplot_spec):
        """Create the controls to show the original properties."""
        data = []
        bounds = []
        for i in range(len(self.cont_prop)):
            bounds.append(self.mecha.get_prop_bounds(i+len(self.disc_prop)))
            data.append(
                (i, {'valmin': .5*bounds[-1][0],
                     'valmax': 1.5*bounds[-1][1],
                     'valinit': self.cont_prop[i],
                     'label': self.labels[i],
                     'color': 'black'
                     })
                )
        cp = ControlPane(self.fig, data, None, subplot_spec, bounds=bounds,
                         show_value=False)
        for s in cp.sliders.values():
            s.active = False
            s.eventson = False
            s.drawon = False

        return cp

    def create_options(self, ax):
        """Create the option buttons."""
        opt = DemoOptions(ax, self)
        # Hide ref plot.
        b_hide_crv = Button(self.fig.add_axes([.15, .05, .1, .05]),
                            "Hide ref. plot")
        b_hide_crv.on_clicked(opt.hide_ref)
#        # Show mecha.
#        b_show_mecha = Button(self.fig.add_axes([.3, .05, .1, .05]),
#                              "Show mechanism")
#        b_show_mecha.on_clicked(opt.show_mecha)
        # Show before/after.
        b_show_ba = Button(self.fig.add_axes([.3, .05, .1, .05]),
                            "Show before/after")
        b_show_ba.on_clicked(opt.show_before_after)

        return opt,  (b_hide_crv, b_show_ba)

    def redraw(self):
        """Redraw dynamic elements."""
        self.new_crv_plt.set_data(self.new_crv[0], self.new_crv[1])
        if self.new_poi is not None:
            self.new_poi_plt.set_offsets([self.new_poi.T])
            self.new_poi_plt.set_visible(True)
        else:
            self.new_poi_plt.set_visible(False)
#        if self.options.state['show_mecha']:
#            self.mecha_plt.redraw()

        # First solution: clean, simple, but slow.
        self.fig.canvas.draw_idle()

        # Second solution: faster, with small artifacts.
        # Actually more artifacts are introduced than shown, but they are
        # hidden because we blit in a limited region. However, unfocusing
        # then refocusing the window will reveal them all. Resizing the window
        # will remove them, though.
#        self.ax[0].redraw_in_frame()
#        self.fig.canvas.blit(self.ax[0].bbox.translated(1,0))
#        self.ax[1].redraw_in_frame()
#        self.fig.canvas.blit(self.ax[1].bbox.translated(1,0))
#
#        for slider in self.control.sliders.values():
#            slider.ax.redraw_in_frame()
#            self.fig.canvas.blit(
#                slider.ax.bbox.translated(1,-.5).expanded(1,1.05))
#        for slider in self.ref_control.sliders.values():
#            slider.ax.redraw_in_frame()
#            self.fig.canvas.blit(
#                slider.ax.bbox.translated(1,-.5).expanded(1,1.05))

        # Third solution is much more involved, but works perfectly. See
        # http://matplotlib.org/examples/event_handling/path_editor.html
        # for an example.

    ### CONTROLLER

    def set_cont_prop(self, props, update_bounds=True):
        """Set the continuous property vector, update data."""
        self.cont_prop = props
#        print(props)
        # We need to update all the parameters before getting the bounds.
        self.mecha.reset(*np.r_[self.disc_prop, props])
        # Update ref sliders.
        for i, prop in enumerate(props):
            self.ref_control.set_val(i, prop)
            if update_bounds:
                bnds = self.mecha.get_prop_bounds(i+len(self.disc_prop))
                self.ref_control.set_bounds(i, bnds)
        # Update new curve and PoI.
        self.new_crv = self.mecha.get_curve(self.nb_crv_pts)
        new_poi, new_par = self.get_corresp(
            self.ref_crv, self.ref_par, [self.new_crv])
        self.new_poi = new_poi[0]
        self.new_par = new_par[0]

    def on_move(self, event):
        """Callback function for mouse move."""
        raise NotImplementedError

    def on_button_press(self, event):
        """Callback function for mouse button press."""
        raise NotImplementedError

    def on_button_release(self, event):
        """Callback function for mouse button release."""
        raise NotImplementedError

    def on_slider_update(self, id_, value):
        """Callback function for slider update."""
        raise NotImplementedError

    def run(self):
        plt.ioff()
        plt.show()


def interp2d(x, y, z, gridsize=(200, 200)):
    """Interpolate data points on a grid."""
    # Define grid.
    xi = np.linspace(x.min(), x.max(), gridsize[0])
    yi = np.linspace(y.min(), y.max(), gridsize[1])
    # Interpolate values.
    xi, yi = np.meshgrid(xi, yi)
    zi = interp.griddata((x, y), z, (xi, yi), method='cubic')
    return xi, yi, zi


def get_highest_quantile(vals, q=50):
    """Returns the indexes of the q-quantile of highest values."""
    imax = int(len(vals) / q)
    return np.argpartition(-vals, imax)[:imax]


def fit_poly(s, d=2, w=None):
    """Apply polynomial least-squares fitting to the input samples.

    Weights, if given, should be positive.
    """
    c = npol.polyfit(s[:, 0], s[:, 1], d, w=w)
    # polyfit returns coeffs in increasing powers, while poly1d expects them
    # in decreasing powers.
    c = c[::-1]
    return np.poly1d(c)


class TwoDimsDemo(InvarDemo):
    """Specialization for mechanisms with 2 continuous properties.

    Attributes
    ----------
    deg_invar_poly: int
        Degree of the polynomial used to fit the invariant space.
    samples: N_samples x N_cont_props numpy array
        Array of valid property samples (for display).
    scores: N_samples numpy array
        Array of invariance scores corresponding to each sample.
    opt_path: N_cont_props x pts_per_dim numpy array
        Sequence of points optimally close to the solution space.
    phi_opt: callable
        Interpolation of the previous path.
    """
    def __init__(self, disc_prop, cont_prop, init_poi_id, pts_per_dim=20,
                 keep_ratio=.05, deg_invar_poly=2, nb_crv_pts=2**6):
        # Initial parameters.
        self.disc_prop = disc_prop
        self.cont_prop = cont_prop
        self.pts_per_dim = pts_per_dim
        self.keep_ratio = keep_ratio
        self.deg_invar_poly = deg_invar_poly
        self.mecha = EllipticSpirograph(*self.disc_prop+self.cont_prop)
        self.labels = ["$e^2$", "$d$"]
        self.nb_crv_pts = nb_crv_pts
        # Reference curve and parameter.
        self.ref_crv = self.mecha.get_curve(self.nb_crv_pts)
        self.ref_par = init_poi_id
        self.ref_poi, self.ref_par = self.get_corresp(
            self.ref_crv, self.ref_par, [self.ref_crv])
        self.ref_poi, self.ref_par = self.ref_poi[0], self.ref_par[0]
        # New curve and parameter(s).
        self.new_crv = None
        self.new_poi = None
        # Solution space.
        self.samples = None
        self.scores = None
        self.phi = None
        self.invar_space_bnds = None
        # Optimal solution.
        self.opt_path = None
        self.phi_opt = None

        self.compute_invar_space()

        self.init_draw()

    ### MODEL

    def sample_props(self, nb=5, extent=1.):
        bnd_p1 = self.mecha.get_prop_bounds(2)
        eps = 2 * self.mecha.constraint_solver.eps

        for p1 in np.linspace(bnd_p1[0], bnd_p1[1] - eps, nb):
            bnd_p2 = self.mecha.constraint_solver.get_bounds(
                self.disc_prop+(p1, None), 3)
            for p2 in np.linspace(bnd_p2[0], bnd_p2[1] - eps, nb):
                yield p1, p2

    def compute_invar_space(self):
        self.samples = np.array(list(self.sample_props(self.pts_per_dim)))
        self.scores = self.get_invar_scores(self.samples)
        # Filter out low scores and fit curve.
        ids = get_highest_quantile(self.scores, q=1/self.keep_ratio)
        if len(ids) < self.deg_invar_poly + 1:
            print("Warning: too few samples for the "
                  "regression ({})".format(len(ids)))
        self.phi = fit_poly(self.samples[ids], w=self.scores[ids],
                             d=self.deg_invar_poly)
        # Redefine bounds.
        self.bnds_invar_space = (self.samples[ids, 0].min(),
                                 self.samples[ids, 0].max())
        # Optimal solution.
        self.opt_path = np.asarray(self.get_optimal_path())
        self.phi_opt = interp.interp1d(*self.opt_path)

    ### VIEW

    def init_draw(self):
        """Initialize canvas."""
        super().init_draw()

        self.new_crv_pos = None
        self.draw_prop_space(self.ax[1])

    def draw_prop_space(self, frame):
        """Draw the 2D property space, the samples, and the reference point."""
        frame.margins(0.1)
        frame.set_xlabel(self.labels[0])
        frame.set_ylabel(self.labels[1])
        frame.set_title("Continuous property space (hidden).\n"
                        "The invariant subspace is computed.\n")
        samples = self.samples.T
        # Draw the boundary.
        frame.add_patch(self.get_bound_poly())
        # Draw the interpolated scores.
        xi, yi, zi = interp2d(*samples, z=self.scores)
        zi = np.ma.masked_invalid(zi)
        pcol = frame.pcolormesh(xi, yi, zi, cmap=self.cmap)
        # Draw the samples.
        frame.scatter(*samples, marker='.', c='k', linewidth=.1,
                      label="Sample")
        # Draw the optimal solution.
        frame.plot(*self.opt_path, color='m', linestyle='dashed',
                   linewidth=4, label="Optimal sol.")
        # Draw the approximate solution.
        frame.plot(*self.get_approx_path(), color='c', linewidth=2,
                   label="Approximate sol.")
        # Draw the position of the reference curve.
        frame.scatter(
            *self.cont_prop, s=300, c='lightyellow', marker='*',
            edgecolor=self.ref_crv_plt.get_color(), zorder=3,
            label="Ref. curve")
        # Draw the position of the new curve.
        self.new_crv_pos = frame.scatter(
            [], [], s=300, c=self.new_crv_plt.get_color(), marker='*',
            edgecolor='none', zorder=3,label="New curve")
        # Draw the legend.
        # Dummy plot to avoid the ugly patch symbol of the boundary.
        frame.plot([], [], c='r', linewidth=2, label="Boundary")
        frame.legend(loc='upper left', scatterpoints=1, ncol=2,
                     fontsize='medium')
        # Draw the colorbar.
        bounds = (self.scores.min(), self.scores.max())
        cbar = self.fig.colorbar(pcol, ax=frame, ticks=bounds, format='%.2f')
        cbar.ax.set_ylabel("Invariance score", rotation=270)

    def get_bound_poly(self):
        """Return the boundary of the feasible space as a Patch."""
        bnd_p1 = self.mecha.get_prop_bounds(2)
        pts_top, pts_btm = [], []
        for p1 in np.linspace(bnd_p1[0], bnd_p1[1], self.pts_per_dim):
            bnd_p2 = self.mecha.constraint_solver.get_bounds(
                self.disc_prop+(p1, .2), 3)
            pts_btm.append((p1, bnd_p2[0]))
            pts_top.append((p1, bnd_p2[1]))
        return Polygon(pts_btm+pts_top[::-1], alpha=0.9, facecolor='none',
                       edgecolor='r', linewidth=2)

    def get_optimal_path(self):
        """Return the invariant space computed optimally."""
        bnd_p1 = self.bnds_invar_space
        p1 = np.linspace(*bnd_p1, num=self.pts_per_dim)
        init = self.mecha.props.copy()
        ref_ft = self.get_features(self.ref_crv, self.ref_par, self.ref_poi)
#        print("Ref. feature", ref_ft)
        p2 = []
        for val in p1:
            self.mecha.update_prop(2, val)
            def obj_func(x):
                self.mecha.update_prop(3, x[0])
                crv = self.mecha.get_curve(self.nb_crv_pts)#[:, :-1]
                poi, par = self.get_corresp(self.ref_crv, self.ref_par, [crv])
                ft = self.get_features(crv, par[0], poi[0])
                diff = ref_ft - ft
                return np.dot(diff, diff)
            bnd_p2 = list(self.mecha.constraint_solver.get_bounds(
                self.disc_prop+(val, None), 3))
            # Adjust upper bound (solver tends to exceed it slightly).
            bnd_p2[1] -= 1e4 * self.mecha.constraint_solver.eps
            p2.append(
                opt.minimize(
                    obj_func, init[3], method='L-BFGS-B', bounds=[bnd_p2]).x)
#                opt.fsolve(obj_func, init[3]))
        self.mecha.reset(*init)
        return p1, p2

    def get_approx_path(self):
        """Return the estimated invariant space."""
        p1 = np.linspace(*self.bnds_invar_space, num=self.pts_per_dim)
        p2 = self.phi(p1)
        # TODO: make it clean, call get_prop_bounds
        valid = np.logical_and(self.samples[:, 1].min() <= p2,
                               p2 <= self.samples[:, 1].max())
        return p1[valid], p2[valid]

    def create_controls(self, subplot_spec):
        """Create the controls to explore the invariant space."""
        bounds = self.bnds_invar_space
        data = (
            ('app', {'valmin': bounds[0],
                     'valmax': bounds[1],
                     'valinit': self.cont_prop[0],
                     'label': "Approx.\nsolution",
                     'color': 'c'
                    }),
            ('opt', {'valmin': bounds[0],
                     'valmax': bounds[1],
                     'valinit': self.cont_prop[0],
                     'label': "Optimal\nsolution",
                     'color': 'm'
                    })
            )
        return ControlPane(self.fig, data, self.on_slider_update, subplot_spec,
                           show_value=False)

    def redraw(self):
        """Redraw dynamic elements."""
        self.new_crv_pos.set_offsets([self.mecha.props[2],
                                      self.mecha.props[3]])
        super().redraw()

    ### CONTROLLER

    def on_slider_update(self, id_, value):
        """Callback function for slider update."""
        if id_ == 'app':
            phi = self.phi
        elif id_ == 'opt':
            phi = self.phi_opt

        cont_prop = (value, phi(value))
        print("New continuous properties: {}".format(cont_prop))
        self.set_cont_prop(cont_prop)

        self.redraw()


class ManyDimsDemo(InvarDemo):
    """Specialization for mechanisms with 2 continuous properties or more.

    Attributes
    ----------
    slider_active: bool
        Used to know if the slider is being used.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.init_draw()

        # Controller
        self.slider_active = False
        self.fig.canvas.mpl_connect('button_release_event',
                                    self.on_button_release)

    ### VIEW

    def init_draw(self):
        super().init_draw()

        self.mecha_plt = None
        self.draw_plt = None
        self.draw_machine(self.ax[1])

    def draw_machine(self, frame):
        frame.margins(0.1)
        frame.set_xticks([])
        frame.set_yticks([])
        frame.set_axis_bgcolor('.9')
#        frame.margins(.1)
        frame.set_aspect('equal')
        frame.set_title("Drawing machine (hidden).\n")
        self.draw_plt = frame.plot(*self.ref_crv, lw=1, alpha=.8)[0]
        self.mecha_plt = mechaplot_factory(self.mecha, frame, self.draw_plt)
        self.mecha_plt.redraw()

    def redraw(self):
        self.draw_plt.set_data(*self.new_crv)
        self.mecha_plt.redraw()
        super().redraw()

    ### CONTROLLER

    def create_controls(self, subplot_spec):
        """Create the controls to explore the invariant space."""
        data = [
            (i, {'valmin': -1.,
                 'valmax': 1.,
                 'valinit': self.cont_prop_invar_space[i],
                 'label': "$x_{}$".format(i+1)
                 })
            for i in range(self.ndim_invar_space)
            ]

        return ControlPane(self.fig, data, self.on_slider_update, subplot_spec,
                           bounds=self.bnds_invar_space, show_value=False)

    def on_button_release(self, event):
        """Callback function for mouse button release."""
        if self.slider_active:
            self.slider_active = False

            cont_prop = self.project_cont_prop_vect()
            self.set_cont_prop(cont_prop)
            self.compute_invar_space()
            # Update sliders.
            for i, val in enumerate(self.cont_prop_invar_space):
                self.control.set_val(i, val, incognito=True)
                self.control.set_bounds(i, self.bnds_invar_space[i])

            self.redraw()

    def on_slider_update(self, id_, value):
        """Callback function for slider update."""
        self.slider_active = True

        self.cont_prop_invar_space[id_] = value
        cont_prop = self.phi(self.cont_prop_invar_space).ravel()
        self.set_cont_prop(cont_prop, update_bounds=False)
        # Update slider bounds.
#        for i in range(self.ndim_invar_space):
#            if i != id_:
#                bnds = self.get_bounds_invar_space(i)
#                self.control.set_bounds(i, bnds)

        self.redraw()
