# -*- coding: utf-8 -*-
"""
Spyrou, the drawing-machine designer.

@author: Robin Roussel
"""
from collections import OrderedDict
from enum import Enum
import math
import textwrap

from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

from controlpane import ControlPane, make_slider
from mecha import EllipticSpirograph#, SingleGearFixedFulcrumCDM
from pois import find_krv_max, find_isect
import curvedistances as cdist

Modes = Enum('Modes', 'sketcher editor')
Actions = Enum('Actions',
               'none sketch set_min_bound set_max_bound set_sym search show')
Invariants = Enum('Invariants',
                  'position curvature intersection_angle distance on_line')


class FancyButton(Button):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ax = self.ax
        ax.set_axis_bgcolor('none')
        ratio = ax.bbox.width / ax.bbox.height
        ax.set_xlim(0, ratio)
        for s in ax.spines.values():
            s.set_color('none')
        pad = .15
        shape = FancyBboxPatch((pad, pad), ratio-2*pad, (1-2*pad),
                               fc=self.color, ec='none',
                               boxstyle="round,pad={}".format(pad))
        ax.add_patch(shape)

    def _motion(self, event):
        if self.ignore(event):
            return
        if event.inaxes == self.ax:
            c = self.hovercolor
        else:
            c = self.color
        if c != self._lastcolor:
            self.ax.patches[0].set_facecolor(c)
            self._lastcolor = c
            if self.drawon:
                self.ax.figure.canvas.draw()


class CheckButton(Button):

    def __init__(self, *args, **kwargs):
        self.pushcolor = kwargs.pop('pushcolor', None)
        super().__init__(*args, **kwargs)
        self.pushed = False
        self.on_clicked(self.push)

    def push(self, event):
        self.pushed = not self.pushed
        if self.pushed:
            self.ax.set_axis_bgcolor(self.pushcolor)
        else:
            self.ax.set_axis_bgcolor(self._lastcolor)
        if self.drawon:
            self.ax.figure.canvas.draw()

    def _motion(self, event):
        if self.ignore(event):
            return
        if self.pushed:
            return
        if event.inaxes == self.ax:
            c = self.hovercolor
        else:
            c = self.color
        if c != self._lastcolor:
            self.ax.set_axis_bgcolor(c)
            self._lastcolor = c
            if self.drawon:
                self.ax.figure.canvas.draw()

class Model:
    # Dictionaries used to build each invariant. If the type is present in the
    # dict, it means that it is supported by the Model.
    krv_invar = OrderedDict(((Invariants.position, False),
                             (Invariants.curvature, False),
                             (Invariants.on_line, False)))
    isect_invar = OrderedDict(((Invariants.position, False),
                               (Invariants.intersection_angle, False),
                               (Invariants.on_line, False)))

    def __init__(self):
        self.crv_bnds = [None, None]
        self.sym_order = 1
        self.strokes = [] # List of N*2 lists of points
        self.undone_strokes = []
        # Mechanism retrieval
        self.samples = self.get_global_sampling()
        self.distance = cdist.DistanceField().get_dist
        # List of dicts with keys ('type', 'props', 'curve')
        self.search_res = []
        self.mecha = None
        self.dprops = []
        self.cprops = []
        self.nb_cp = 2
        self.nb_dp = 2
        self.cbounds = []
        self.crv = None
        # The following are private; PoIs should be accessed with get_poi.
        self._krv_pois = None
        self._isect_pois = None
        self._poi_dict = {}
        # Dictionary of poi_key:{Invariance:bool, ...} pairs
        self.invar_dict = {}

    def compute_pois(self):
        self._krv_pois = find_krv_max(self.crv)
        self._isect_pois = find_isect(self.crv)

    def reset_pois_dict(self):
        """Regenerate the PoI dictionary.

        The PoI dict maps a unique key to a position in the corresponding
        PoI container.

        This method should be called each time the mechanism or the discrete
        properties change, and only after compute_pois has been called.

        Each PoI has a key that is given to exterior classes. This way, even
        if the features of the PoI change, the PoI key is still valid. In other
        words, each key 'follows' its PoI when we explore the parameter space.
        """
        # We assume that there won't be more than 999 PoIs.
        if self._krv_pois is not None and self._krv_pois.size:
            self._poi_dict = {'_krv_pois{:03d}'.format(i):i
                             for i in range(self._krv_pois.shape[0])}
            shift = self._krv_pois.shape[0]
        else:
            self._poi_dict = {}
            shift = 0
        if self._isect_pois is not None and self._isect_pois.size:
            shift = self._isect_pois.shape[0]
            self._poi_dict.update(
                {'_isect_pois{:03d}'.format(j+shift):j
                 for j in range(self._isect_pois.shape[0])}
                )

    def find_close_pois_keys(self, position, radius):
        norm = np.linalg.norm
        return [key for key, val in self._poi_dict.items()
                if norm(getattr(self, key[:-3])[val, :2] - position) <= radius]

    def get_poi(self, key):
        value = self._poi_dict.get(key)
        if value is None:
            return None
        return getattr(self, key[:-3])[value]

    def get_poi_invariance(self, key):
        """Get the current invariance state of a PoI, as an Invariance:state
        orderred dictionary.

        Do not use this function to set the invariance state of a PoI:
        changes will not be propagated.
        """
        value = self.invar_dict.get(key)
        if value is None:
            return (Model.krv_invar.copy() if key.startswith('_krv')
                    else Model.isect_invar.copy())
        else:
            return value.copy()

    def set_poi_invariance(self, key, invar):
        """Set the current invariance state of a PoI.

        'invar' should be a Invariance:bool dict.
        """
        value = self.invar_dict.get(key)
        if value is None:
            base = (Model.krv_invar.copy() if key.startswith('_krv')
                    else Model.isect_invar.copy())
            base.update(invar)
            self.invar_dict[key] = base
        else:
            value.update(invar)

    def get_global_sampling(self):
        """Sample feasible parameters across all mechanisms."""
        samples = {}
        mechanisms_types = (EllipticSpirograph, )#SingleGearFixedFulcrumCDM)
        for type_ in mechanisms_types:
            samples[type_] = np.array(list(
                type_.ConstraintSolver.sample_feasible_domain(
                    grid_resol=(4,)*4)))
        return samples

    def search_mecha(self, nb):
        if not len(self.strokes):
            return
        sketch = np.hstack([np.array(stroke).T for stroke in self.strokes])
        self.search_res.clear()
        ranges = [0]
        # Convert types and samples to lists to keep the order.
        types = list(self.samples.keys())
        samples = list(self.samples.values())
        # Pre-filter the samples.
        if self.sym_order > 1:
            samples = [s[s[:, 0] == self.sym_order] for s in samples]
        distances = []
        for type_, type_samples in zip(types, samples):
            ranges.append(ranges[-1] + type_samples.shape[1])
            mecha = type_(*type_samples[0])
            for sample in type_samples:
                mecha.reset(*sample)
                crv = mecha.get_curve()
                distances.append(self.distance(crv, sketch))
        print(len(distances))
        best = np.argpartition(np.array(distances), nb)[:nb]
        ranges = np.array(ranges)
        for id_ in best:
            # Find index in ranges with a small trick: argmax gives id of the
            # first max value, here True.
            typeid = np.argmax(ranges > id_) - 1
            type_ = types[typeid]
            mecha = type_(*samples[typeid][id_-ranges[typeid]])
            self.search_res.append({
                'type': type(mecha),
                'props': mecha.props.copy(),
                'curve': mecha.get_curve()
                })

    def set_mecha(self, mecha):
        self.mecha = mecha
        self.crv = self.mecha.get_curve()
        self.dprops = self.mecha.props[:self.nb_dp]
        self.cprops = self.mecha.props[self.nb_dp:]
        self.cbounds = [self.mecha.get_prop_bounds(i+self.nb_dp)
                        for i, prop in enumerate(self.cprops)]
        self.compute_pois()
        self.reset_pois_dict()

    def set_cont_prop(self, props):
        """Set the continuous property vector, update data."""
        self.cprops = props
        # We need to update all the parameters before getting the bounds.
        self.mecha.reset(*np.r_[self.dprops, props])
        self.cbounds = [self.mecha.get_prop_bounds(i+self.nb_dp)
                        for i, prop in enumerate(props)]
        # Update curve and PoI.
        self.crv = self.mecha.get_curve()


class View:

    def __init__(self, fig_size=(12,8), start_mode=Modes.sketcher):
        self.fig = plt.figure(figsize=fig_size, facecolor='.2')
        self.fig.canvas.set_window_title('Spyrou2.0')
        plt.subplots_adjust(left=0., right=1., bottom=0., top=1.,
                            wspace=0., hspace=0.)
        self.grid_size = (3*fig_size[1], 3*fig_size[0])
        self.colors = {
            'light': '.9',
            'bg': '.2'
            }

        self.sk_canvas = self.draw_sketch_canvas()
        self.ed_canvas = self.draw_edit_canvas()
        self.hide_layer([self.ed_canvas])

        self.widgets = {}

        self.sk_head = None
        self.sk_layer = []
        self.draw_sketcher_tab()

        self.max_nb_props = 5
        self.nb_props = 0
        self.max_nb_cstr_per_poi = 3

        self.ed_head = None
        self.ed_layer = []
        self.controls = None
        self.draw_editor_tab()
        self.update_controls([], [], redraw=False)
        self.update_invariants([], [], redraw=False)
        self.hide_layer(self.ed_layer)

        self.draw_bottom_pane()

        self.overlay = self.draw_overlay()
        self.hide_layer([self.overlay])

        # Wait until first call of show_search_menu to instantiate axes.
        self.se_layer = []

        self.undone_plots = []
        self.borders = [None, None]
        self.sym_lines = []
        self.selected_poi = None
        self.locked_pois = []

    @staticmethod
    def remove_axes(ax):
        """Remove the actual 'axes' from a matplotlib Axes object."""
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        for s in ax.spines.values():
            s.set_color('none')
        # For the record, here's the more detailed version:
#        ax.spines['bottom'].set_color('none')
#        ax.spines['top'].set_color('none')
#        ax.spines['right'].set_color('none')
#        ax.spines['left'].set_color('none')


    ### Low-level graphical elements (buttons, separators, etc.)

    def draw_sketch_canvas(self):
        canvas = plt.subplot2grid(
            self.grid_size, (0, 0), rowspan=self.grid_size[0],
            colspan=self.grid_size[0])
        self.remove_axes(canvas)
        canvas.set_xlim([-1, 1])
        canvas.set_ylim([-1, 1])
        canvas.set_aspect('equal', 'datalim')
        return canvas

    def draw_edit_canvas(self):
        canvas = self.fig.add_axes(self.sk_canvas.get_position())
        self.remove_axes(canvas)
#        canvas.set_xlim([-1, 1])
#        canvas.set_ylim([-1, 1])
        canvas.margins(.1)
        canvas.set_aspect('equal', 'datalim')
        return canvas

    def draw_tab_header(self, grid_pos, label, active=False):
        width = (self.grid_size[1] - self.grid_size[0]) // 2
        tab_ax = plt.subplot2grid(
            self.grid_size, grid_pos, rowspan=2, colspan=width)
        self.remove_axes(tab_ax)
        if active:
            tab = Button(tab_ax, label, color='.2', hovercolor='.2')
        else:
            tab = Button(tab_ax, label, color='.4', hovercolor='.4')
        tab.label.set_fontsize(16)
        tab.label.set_color('.9')
        tab.label.set_weight('light')
        return tab

    def draw_section_title(self, grid_pos, label):
        width = (self.grid_size[1] - self.grid_size[0]) // 2
        ax = plt.subplot2grid(
            self.grid_size, grid_pos, rowspan=1, colspan=width, axisbg='none')
        self.remove_axes(ax)
        ax.set_navigate(False)
        title = ax.text(0.5, 0.5, label,
                        verticalalignment='bottom',
                        horizontalalignment='center',
                        transform=ax.transAxes)
        title.set_fontsize(15)
        title.set_weight('light')
        title.set_color('.9')
        return ax

    def draw_slider(self, grid_pos, slider_args, width):
        ax = plt.subplot2grid(
            self.grid_size, grid_pos, rowspan=1, colspan=width, axisbg='.9')
        self.remove_axes(ax)
        slider_args['color'] = 'lightgreen'
        slider = make_slider(ax, **slider_args)
        slider.label.set_weight('bold')
        slider.label.set_color('.9')
        slider.label.set_x(-0.25)
        slider.label.set_horizontalalignment('center')
        slider.valtext.set_weight('bold')
        slider.valtext.set_color('.9')
        return slider

    def draw_button(self, grid_pos, width, height, label):
        bt_ax = plt.subplot2grid(
            self.grid_size, grid_pos, rowspan=height, colspan=width)
        bt = FancyButton(bt_ax, label, color='.9', hovercolor='lightgreen')
        bt.label.set_fontsize(12)
        bt.label.set_weight('bold')
        bt.label.set_color('.2')
        return bt

    def draw_check_button(self, grid_pos, width, height, label):
        ax = plt.subplot2grid(
            self.grid_size, grid_pos, rowspan=2, colspan=width)
        for s in ax.spines.values():
            s.set_color('.4')
#            s.set_linewidth(10)
        bt = CheckButton(ax, label, color='.2', hovercolor='.2',
                         pushcolor='.3')
        bt.label.set_fontsize(12)
        bt.label.set_weight('bold')
        bt.label.set_color('.9')
        return bt

    def draw_separator(self, grid_pos):
        width = (self.grid_size[1] - self.grid_size[0])
        ax = plt.subplot2grid(
            self.grid_size, grid_pos, rowspan=1, colspan=width, axisbg='none')
        self.remove_axes(ax)
        ax.set_navigate(False)
        ax.axhline(.1, 0.01, .99, color='.5', linestyle='dashed')
        return ax

    def draw_overlay(self):
        ax = self.fig.add_axes([0, 0, 1, 1])
        self.remove_axes(ax)
        ax.set_navigate(False)
        ax.patch.set_alpha(.5)
        return ax

    def draw_search_cell(self, coords, id_, drawing=None):
        ax = self.fig.add_axes(coords, axisbg='.9', zorder=2)
        ax.set_aspect('equal', 'datalim')
        ax.margins(.1)
        for s in ax.spines.values():
            s.set_color('.2')
            s.set_linewidth(10)
        if drawing is not None:
            ax.plot(*drawing)
        else:
            ax.plot([], [])
        bt = Button(ax, '', color='.9', hovercolor='lightgreen')
        # Attach an id to the Axes so that we can identify it when clicked.
        ax.cell_id = id_
        return bt

    def draw_inner_bound(self, radius):
        self.borders[0] = Circle((0, 0), radius, fc='.9', ec='.7', ls='dashed')
        self.sk_canvas.add_patch(self.borders[0])

    def draw_outer_bound(self, radius):
        self.sk_canvas.set_axis_bgcolor('.9')
        self.borders[1] = Circle(
            (0, 0), radius, fc='1.', ec='.7', ls='dashed', zorder=-1)
        self.sk_canvas.add_patch(self.borders[1])

    def draw_sym_lines(self, order):
        if order > 1:
            radius = 2 * math.sqrt(
                max(np.abs(self.sk_canvas.get_xlim()))**2 +
                max(np.abs(self.sk_canvas.get_ylim()))**2
                )
            for i in range(order):
                angle = 2 * math.pi * i / order
                self.sym_lines.append(
                    self.sk_canvas.plot(
                        (0., radius*math.cos(angle)),
                        (0., radius*math.sin(angle)),
                        c='.7', ls='dashed')[0]
                    )
            # Move these plots at the beginning so that they don't disturb the
            # undo/redo.
            self.sk_canvas.lines = (self.sk_canvas.lines[-order:] +
                                 self.sk_canvas.lines[:-order])

    def draw_controls(self, grid_pos, data, bounds, width):
        gs = GridSpec(*self.grid_size)
        sub_spec = gs[grid_pos[0]:grid_pos[0]+len(data),
                      grid_pos[1]:grid_pos[1]+width]
        for item in data:
            item[1]['color'] = 'lightgreen'
            item[1]['alpha'] = 1
        cp = ControlPane(self.fig, data, lambda x: x, sub_spec, bounds,
                         show_value=False)
        for slider in cp.sliders.values():
            ax = slider.ax
            ax.set_axis_bgcolor('.9')
            ax.spines['bottom'].set_color('.2')
            ax.spines['bottom'].set_linewidth(5)
            ax.spines['top'].set_color('.2')
            ax.spines['top'].set_linewidth(5)
            ax.spines['right'].set_color('none')
            ax.spines['left'].set_color('none')
            if slider.slidermin is not None:
                slider.slidermin.poly.set_color('orange')
            if slider.slidermax is not None:
                slider.slidermax.poly.set_color('orange')
        return cp

    ### High-level graphical elements (tabs and panels)

    def draw_sketcher_tab(self):
        tab_width = (self.grid_size[1] - self.grid_size[0]) // 2
        row_id = 0

        self.sk_head = self.draw_tab_header(
            (row_id, self.grid_size[0]), "Sketcher", active=True)
        row_id += 3

        self.sk_layer.append(
            self.draw_section_title((row_id, self.grid_size[0]+tab_width//2),
                                    "Construction lines")
            )
        row_id += 1

        self.widgets['sk_bnd'] = self.draw_button(
            (row_id, self.grid_size[0]+tab_width//2), tab_width, 1,
            "Set boundaries")
        self.sk_layer.append(self.widgets['sk_bnd'].ax)
        row_id += 2

        slider_args = {'valmin': 0, 'valmax': 25, 'valinit': 1,
                       'label': "Symmetry\norder"}
        self.widgets['sk_sym'] = self.draw_slider(
            (row_id, self.grid_size[0]+tab_width*3//4), slider_args,
            tab_width*5//4)
        self.sk_layer.append(self.widgets['sk_sym'].ax)
        row_id += 1

        self.sk_layer.append(
            self.draw_separator((row_id, self.grid_size[0]))
            )
        row_id += 2

        self.sk_layer.append(
            self.draw_section_title((row_id, self.grid_size[0]+tab_width//2),
                                    "Sketch")
            )
        row_id += 1

        self.widgets['sk_undo'] = self.draw_button(
            (row_id, self.grid_size[0]+tab_width//4), tab_width*3//4, 2,
            "Undo\nlast stroke")
        self.sk_layer.append(self.widgets['sk_undo'].ax)
        self.widgets['sk_redo'] = self.draw_button(
            (row_id, self.grid_size[0]+tab_width*5//4), tab_width*3//4, 2,
            "Redo\nlast stroke")
        self.sk_layer.append(self.widgets['sk_redo'].ax)

    def draw_editor_tab(self):
        tab_width = (self.grid_size[1] - self.grid_size[0]) // 2
        row_id = 0
        # Put material in sketcher tab aside so that there is not conflict
        # when instantiating axes. Yes, it's ugly, but it works.
        self.move_layer_horizontally(self.sk_layer, -2*tab_width)

        self.ed_head = self.draw_tab_header(
            (row_id, self.grid_size[0]+tab_width), "Editor")
        row_id += 3

        self.ed_layer.append(
            self.draw_section_title((row_id, self.grid_size[0]+tab_width//2),
                                    "Invariants")
            )
        row_id += 1

        for i in range(self.max_nb_cstr_per_poi):
            name = 'ed_invar_{}'.format(i)
            self.widgets[name] = self.draw_check_button(
                (row_id, self.grid_size[0]+tab_width*i*2//3), tab_width*2//3,
                1, '')
            self.ed_layer.append(self.widgets[name].ax)
        row_id += 3

        self.widgets['ed_apply_inv'] = self.draw_button(
            (row_id, self.grid_size[0]+tab_width//2), tab_width, 1,
            "Apply invariants")
        self.ed_layer.append(self.widgets['ed_apply_inv'].ax)
        row_id += 1

        self.ed_layer.append(
            self.draw_separator((row_id, self.grid_size[0]))
            )
        row_id += 2

        self.ed_layer.append(
            self.draw_section_title((row_id, self.grid_size[0]+tab_width//2),
                                    "Controls")
            )
        row_id += 1

        data = []
        bounds = []
        for i in range(self.max_nb_props):
            bounds.append((0., 1.))
            data.append(
                (i, {'valmin': 0.,
                     'valmax': 1.,
                     'valinit': 0.,
                     'label': '',
                     })
                )
        self.controls = self.draw_controls(
            (row_id, self.grid_size[0]+tab_width//4), data, bounds,
            tab_width*7//4)
        for id_, slider in self.controls.sliders.items():
            self.widgets['ed_prop_{}'.format(id_)] = slider
            self.ed_layer.append(slider.ax)

        # Put back the sketcher tab in place.
        self.move_layer_horizontally(self.sk_layer, 2*tab_width)

    def draw_bottom_pane(self):
        width = (self.grid_size[1] - self.grid_size[0]) // 2
        row_id = 16
        self.draw_separator((row_id, self.grid_size[0]))
        row_id += 2
        self.draw_section_title((row_id, self.grid_size[0]+width//2),
                                "Drawing machine")
        row_id += 1
        self.widgets['search'] = self.draw_button(
            (row_id, self.grid_size[0]+width//2), width, 1, "Search database")
        row_id += 2
        self.widgets['show'] = self.draw_button(
            (row_id, self.grid_size[0]+width//4), width*3//4, 2,
            "Show\nmechanism")
        self.widgets['export'] = self.draw_button(
            (row_id, self.grid_size[0]+width*5//4), width*3//4, 2,
            "Export\nmechanism")

    def draw_search_menu(self, results, nb_col=3):
        nb_res = len(results)
        nb_rows = math.ceil(nb_res / nb_col)
        cell_size = .15
        ratio = self.grid_size[1] / self.grid_size[0]
        x0 = .5 - cell_size*nb_col/2
        y0 = .5 + cell_size*ratio*(nb_rows/2 - 1)
        for i, result in enumerate(results):
            x = x0 + (i % nb_col)*cell_size
            y = y0 - (i // nb_col)*cell_size*ratio
            cell = self.draw_search_cell([x, y, cell_size, cell_size*ratio],
                                         i, result)
            self.widgets['se_cell_{}'.format(i)] = cell
            self.se_layer.append(cell.ax)

    ### Update view

    @staticmethod
    def hide_layer(layer):
        for panel in layer:
            panel.set_visible(False)
            panel.set_zorder(-1)

    @staticmethod
    def show_layer(layer, zorder=0):
        for panel in layer:
            panel.set_visible(True)
            panel.set_zorder(zorder)

    @staticmethod
    def move_layer_horizontally(layer, delta):
        for panel in layer:
            pos = panel.get_position()
            pos.x0 += delta
            pos.x1 += delta
            panel.set_position(pos)

    @staticmethod
    def move_layer_vertically(layer, delta):
        for panel in layer:
            pos = panel.get_position()
            pos.y0 += delta
            pos.y1 += delta
            panel.set_position(pos)

    @staticmethod
    def put_header_in_front(head):
        head.color='.2'
        head.hovercolor='.2'
        head.ax.set_axis_bgcolor(head.color)

    @staticmethod
    def put_header_behind(head):
        head.color='.4'
        head.hovercolor='.4'
        head.ax.set_axis_bgcolor(head.color)

    def clear_canvas(self, canvas):
        canvas.lines = []
        canvas.patches = []
        canvas.set_axis_bgcolor('1.')

    def redraw_axes(self, ax):
        ax.redraw_in_frame()
#        self.fig.canvas.blit(ax)
        self.fig.canvas.update()

    def show_search_menu(self, results, redraw=True):
        self.show_layer([self.overlay])
        if self.se_layer:
            for ax, result in zip(self.se_layer, results):
                ax.lines[0].set_data(*result)
            self.show_layer(self.se_layer)
        else:
            self.draw_search_menu(results)
        # The following is more of a Controller thing, but for now it's more
        # convenient to do it here.
        for name, wg in self.widgets.items():
            if name.startswith('se_cell_'):
                wg.set_active(True)
        if redraw:
            self.fig.canvas.draw()

    def hide_search_menu(self, redraw=True):
        self.hide_layer(self.se_layer)
        self.hide_layer([self.overlay])
        # The following is more of a Controller thing, but for now it's more
        # convenient to do it here.
        for name, wg in self.widgets.items():
            if name.startswith('se_cell_'):
                wg.set_active(False)
        if redraw:
            self.fig.canvas.draw()

    def remove_borders(self):
        self.sk_canvas.set_axis_bgcolor('1.')
        for patch in self.borders:
            patch.remove()
        self.borders = [None, None]

    def remove_sym_lines(self):
        for line in self.sym_lines:
            line.remove()
        self.sym_lines = []

    def switch_mode(self, mode, redraw=True):
        sketcher = {
            'head': self.sk_head, 'layer': self.sk_layer,
            'canvas': self.sk_canvas, 'prefix': 'sk_'
            }
        editor = {
            'head': self.ed_head, 'layer': self.ed_layer,
            'canvas': self.ed_canvas, 'prefix': 'ed_'
            }
        if mode is Modes.editor:
            to_show = editor
            to_hide = sketcher
        elif mode is Modes.sketcher:
            to_show = sketcher
            to_hide = editor

        self.put_header_in_front(to_show['head'])
        self.put_header_behind(to_hide['head'])
        self.show_layer(to_show['layer'])
        self.hide_layer(to_hide['layer'])
        self.show_layer([to_show['canvas']])
        self.hide_layer([to_hide['canvas']])
        # The following is more of a Controller thing, but for now it's more
        # convenient to do it here.
        for name, wg in self.widgets.items():
            if name.startswith(to_show['prefix']):
                wg.set_active(True)
            elif name.startswith(to_hide['prefix']):
                wg.set_active(False)

        if redraw:
            self.fig.canvas.draw()

    def awaken_widget(self, label):
        wg = self.widgets.get(label)
        if wg is None:
            # Widget has been killed.
            wg = self.widgets.pop('_'+label)
            self.widgets[label] = wg
            # The tab layers can be repeatedly shown/hidden:
            if label.startswith('ed_'):
                self.ed_layer.append(wg.ax)
            elif label.startswith('sk_'):
                self.ed_layer.append(wg.ax)
            self.show_layer([wg.ax])
            wg.set_active(True)

    def kill_widget(self, label):
        wg = self.widgets.get(label)
        if wg is not None:
            # Widget is alive.
            self.widgets['_'+label] = self.widgets.pop(label)
            # The tab layers can be repeatedly shown/hidden:
            if label.startswith('ed_'):
                self.ed_layer.pop(self.ed_layer.index(wg.ax))
            elif label.startswith('sk_'):
                self.sk_layer.pop(self.sk_layer.index(wg.ax))
            self.hide_layer([wg.ax])
            wg.set_active(False)

    def update_controls(self, data, bounds, redraw=True):
        assert(len(data) == len(bounds))
        self.nb_props = len(data)
        for i in range(self.max_nb_props):
            if i < self.nb_props:
                _, args = data[i]
                self.controls.set_valminmax(i, args['valmin'],
                                            args['valmax'])
                self.controls.set_bounds(i, bounds[i])
                self.controls.set_val(i, args['val'], incognito=True)

                self.awaken_widget('ed_prop_{}'.format(i))
            else:
                self.kill_widget('ed_prop_{}'.format(i))

        if redraw:
            self.fig.canvas.draw()

    def update_editor_plot(self, curve):
        if not len(self.ed_canvas.lines):
            self.ed_canvas.plot(*curve, lw=2, c='b', alpha=.8)
        else:
            self.ed_canvas.lines[0].set_data(*curve)
            self.ed_canvas.relim()
            self.ed_canvas.autoscale()

    def update_pois(self, pois_xy, pois_keys):
        for item in self.ed_canvas.patches:
            if not (item in self.locked_pois or item is self.selected_poi):
                item.remove()

        if pois_xy is not None and len(pois_xy):
            exclude = [poi.poi_key for poi in self.locked_pois]
            if self.selected_poi is not None:
                exclude.append(self.selected_poi.poi_key)

            for xy, key in zip(pois_xy, pois_keys):
                if key not in exclude:
                    self.ed_canvas.add_patch(
                        Circle(xy, radius=.1, fill=True, fc='lightgreen',
                               ec='none', zorder=3, picker=True)
                        )
                    self.ed_canvas.patches[-1].poi_key = key

    def select_poi(self, poi_patch):
        if poi_patch is None or poi_patch is self.selected_poi:
            return
        self.deselect_poi(self.selected_poi)
        self.selected_poi = poi_patch
        poi_patch.set_edgecolor(poi_patch.get_facecolor())
        poi_patch.set_facecolor('none')
        poi_patch.set_linewidth(4)

    def deselect_poi(self, poi_patch):
        if poi_patch is None:
            return
        poi_patch.set_facecolor(poi_patch.get_edgecolor())
        poi_patch.set_edgecolor('none')
        poi_patch.set_linewidth(1)

    def lock_poi(self, poi_patch):
        self.locked_pois.append(poi_patch)
        # PoI being locked is necessarily selected => edgecolor changes
        poi_patch.set_edgecolor('purple')

    def unlock_poi(self, poi_patch):
        self.locked_pois.remove(poi_patch)
        # PoI being unlocked is necessarily selected => edgecolor changes
        poi_patch.set_edgecolor('lightgreen')

    def update_invariants(self, labels, states, redraw=True):
        for i in range(self.max_nb_cstr_per_poi):
            name = 'ed_invar_{}'.format(i)
            if i < len(labels):
                self.awaken_widget(name)
                wg = self.widgets[name]
                title = labels[i].capitalize().replace('_', ' ')
                title = textwrap.fill(title, 12)
                wg.label.set_text(title)
                if wg.pushed != states[i]:
                    drawon = wg.drawon
                    wg.drawon = False
                    wg.push(None)
                    wg.drawon = drawon
            else:
                self.kill_widget(name)

        if redraw:
            self.fig.canvas.draw()


class Controller:

    def __init__(self):
        self.model = Model()
        self.view = View()
        self.connect_widgets()
        self.connect_canvas()

        # Some options
        self.poi_capture_radius = .1 # In normalized coordinates

        # Base state
        self.mode = Modes.sketcher
        self.action = Actions.none

    def run(self):
        plt.ioff()
        plt.show()

    def connect_widgets(self):
        self.view.sk_head.on_clicked(self.select_sk_tab)
        self.view.ed_head.on_clicked(self.select_ed_tab)
        self.view.widgets['sk_bnd'].on_clicked(self.set_sketch_bounds)
        self.view.widgets['sk_undo'].on_clicked(self.undo_stroke)
        self.view.widgets['sk_redo'].on_clicked(self.redo_stroke)
        self.view.widgets['search'].on_clicked(self.search_mecha)
        self.view.widgets['show'].on_clicked(self.show_mecha)
        self.view.widgets['export'].on_clicked(self.export_mecha)
        self.view.widgets['sk_sym'].on_changed(self.set_symmetry)
        self.view.widgets['ed_apply_inv'].on_clicked(self.apply_invariants)
        self.view.controls.update = self.update_mecha_prop

    def connect_canvas(self):
        self.view.fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.view.fig.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.view.fig.canvas.mpl_connect(
            'motion_notify_event', self.on_move)
        self.view.fig.canvas.mpl_connect(
            'pick_event', self.on_pick)

    def check_tb_inactive(self):
        """Check if the matplotlib toolbar plugin is inactive."""
        return self.view.fig.canvas.manager.toolbar._active is None

    def select_sk_tab(self, event):
        if self.mode is Modes.sketcher:
            print("Sketcher mode is already selected.")
        else:
            print("Switching to sketcher mode.")
            self.mode = Modes.sketcher
            self.view.switch_mode(self.mode)

    def select_ed_tab(self, event):
        if self.mode == Modes.editor:
            print("Editor mode is already selected.")
        else:
            print("Switching to editor mode.")
            self.mode = Modes.editor
            self.view.switch_mode(self.mode)

    def set_sketch_bounds(self, event):
        print("Set the bounds of the drawing.")
        self.action = Actions.set_min_bound
        # Flush previous bounds.
        if self.model.crv_bnds[0] is not None:
            self.model.crv_bnds = [None, None]
            self.view.remove_borders()

    def set_symmetry(self, value):
        print("Set the symmetry order of the drawing: {}".format(value))
        self.model.sym_order = value
        self.view.remove_sym_lines()
        self.view.draw_sym_lines(value)

    def undo_stroke(self, event):
        if not len(self.model.strokes):
            print("There is no stroke to undo.")
        else:
            print("Undo the last sketch stroke.")
            self.model.undone_strokes.append(self.model.strokes.pop())
            self.view.undone_plots.append(self.view.sk_canvas.lines.pop())
            self.view.redraw_axes(self.view.sk_canvas)

    def redo_stroke(self, event):
        if not len(self.model.undone_strokes):
            print("There is no undone stroke.")
        else:
            print("Redo the last undone sketch stroke.")
            self.model.strokes.append(self.model.undone_strokes.pop())
            self.view.sk_canvas.lines.append(self.view.undone_plots.pop())
            self.view.redraw_axes(self.view.sk_canvas)

    def search_mecha(self, event):
        if not len(self.model.strokes):
            print("There is no query sketch.")
        else:
            print("Search for the best matching mechanism.")
            self.action = Actions.search
            self.model.search_mecha(6)
            self.view.show_search_menu(
                [res['curve'] for res in self.model.search_res])
            # Since the search cells are lazy-initialized, we have to connect
            # them here if they are instatiated for the first time.
            for name, wg in self.view.widgets.items():
                if name.startswith('se_cell_') and wg.cnt == 0:
                    wg.on_clicked(self.choose_mecha)

    def choose_mecha(self, event):
        # Update controller
        id_ = event.inaxes.cell_id
        print("A mechanism was chosen: {}".format(id_))
        self.action = Actions.none
        self.mode = Modes.editor
        # Update model
        choice = self.model.search_res[id_]
        self.model.set_mecha(choice['type'](*choice['props']))

        # Update view
        self.view.clear_canvas(self.view.ed_canvas)
        self.view.update_editor_plot(choice['curve'])
        data = []
        for i in range(self.model.nb_cp):
            data.append(
                (i, {'valmin': .5*self.model.cbounds[i][0],
                     'valmax': 1.5*self.model.cbounds[i][1],
                     'val': self.model.cprops[i],
                     'label': ''
                     })
                )
        self.view.update_controls(data, self.model.cbounds, redraw=False)
        self.view.hide_search_menu(redraw=False)
        self.view.switch_mode(self.mode, redraw=True)

    def quit_search_mecha(self):
        print("Quit the drawing selection.")
        self.action = Actions.none
        self.view.hide_search_menu()

    def show_mecha(self, event):
        if self.model.mecha is None:
            print("There is no mechanism to show.")
        else:
            print("Show the mechanism.")

    def export_mecha(self, event):
        if self.model.mecha is None:
            print("There is no mechanism to export.")
        else:
            print("Export the mechanism.")

    def add_sketch_point(self, xy, start=False):
        nb_sym = self.model.sym_order
        if start:
            for _ in range(nb_sym):
                # Careful to put this append in the loop, rather than appending
                # N * [], since the latter appends N copies of the same list.
                self.model.strokes.append([])
                self.view.sk_canvas.plot([], [], lw=2, c='b', alpha=.8)
            # Flush history
            self.model.undone_strokes.clear()
            self.view.undone_plots.clear()
        else:
            for i in range(nb_sym):
                angle = 2 * math.pi * i / nb_sym
                cos_, sin_ = math.cos(angle), math.sin(angle)
                xy_ = [xy[0]*cos_ + xy[1]*sin_, -xy[0]*sin_ + xy[1]*cos_]
                self.model.strokes[-nb_sym+i].append(xy_)
                self.view.sk_canvas.lines[-nb_sym+i].set_data(
                    *np.asarray(self.model.strokes[-nb_sym+i]).T)
        self.view.redraw_axes(self.view.sk_canvas)

    def update_sketch_bound(self, radius):
        if self.action is Actions.set_min_bound:
            if self.view.borders[0] is None:
                self.view.draw_inner_bound(radius)
            else:
                self.view.borders[0].radius = radius
        if self.action is Actions.set_max_bound:
            if self.view.borders[1] is None:
                self.view.draw_outer_bound(radius)
            else:
                self.view.borders[1].radius = radius
        self.view.redraw_axes(self.view.sk_canvas)

    def update_mecha_prop(self, id_, value):
        # Update model
        cprops = self.model.cprops
        cprops[id_] = value
        self.model.set_cont_prop(cprops)
        # Update view
        self.view.update_editor_plot(self.model.crv)
        for i, cbounds in enumerate(self.model.cbounds):
            self.view.controls.set_bounds(i, cbounds)

    def find_close_pois(self, position):
        extent = self.view.ed_canvas.get_xlim()
        radius = self.poi_capture_radius * abs(extent[1] - extent[0])
        pois_keys = self.model.find_close_pois_keys(position, radius)
        pois_xy = [self.model.get_poi(key)[:2] for key in pois_keys]
        return pois_xy, pois_keys

    def select_poi(self, event):
        # Change aspect
        self.view.select_poi(event.artist)
        # Change invariants menu
        key = event.artist.poi_key
        invar = self.model.get_poi_invariance(key)
        labels = [type_.name for type_ in invar.keys()]
        states = list(invar.values())
        self.view.update_invariants(labels, states)
        # Update invariants callbacks
        for name, wg in self.view.widgets.items():
            if name.startswith('ed_invar_'):
                invar_type = Invariants[labels[int(name[-1])]]
                if wg.cnt > 1:
                    # Remove previous callback.
                    wg.disconnect(wg.cnt-1)
                wg.on_clicked(self.get_invar_callback(key, invar_type))

    def get_invar_callback(self, key, invar_type):
        def toggle(event):
            print("An invariant was toggled.")
            invar_dict = self.model.get_poi_invariance(key)
            if not True in invar_dict.values():
                self.view.lock_poi(self.view.selected_poi)
            invar_dict[invar_type] = not invar_dict[invar_type]
            if not True in invar_dict.values():
                self.view.unlock_poi(self.view.selected_poi)
            self.model.set_poi_invariance(key, invar_dict)
            self.view.redraw_axes(self.view.ed_canvas)

        return toggle

    def apply_invariants(self, event):
        print("Apply the invariants.")

    def on_move(self, event):
        if self.check_tb_inactive():
            if event.inaxes == self.view.sk_canvas:
                if (self.action is Actions.set_min_bound
                    or self.action is Actions.set_max_bound):
                    radius = math.sqrt(event.xdata**2 + event.ydata**2)
                    self.update_sketch_bound(radius)
                elif self.action is Actions.sketch:
                    self.add_sketch_point([event.xdata, event.ydata])
            elif event.inaxes == self.view.ed_canvas:
                xy, keys = self.find_close_pois((event.xdata, event.ydata))
                self.view.update_pois(xy, keys)
                self.view.redraw_axes(self.view.ed_canvas)

    def on_pick(self, event):
        if self.check_tb_inactive():
            if event.mouseevent.inaxes == self.view.ed_canvas:
                print("A PoI was selected.")
                if event.artist is not self.view.selected_poi:
                    self.select_poi(event)
#                self.view.redraw_axes(self.view.ed_canvas)

    def on_press(self, event):
        if self.check_tb_inactive():
            if event.inaxes == self.view.sk_canvas:
                event.canvas.grab_mouse(self.view.sk_canvas)

                if self.mode is Modes.sketcher:
                    if (self.action is Actions.set_min_bound
                        or self.action is Actions.set_max_bound):
                        pass
                    else:
                        # Default action is sketching.
                        self.action = Actions.sketch
                        self.add_sketch_point([event.xdata, event.ydata],
                                              start=True)

    def on_release(self, event):
        if self.check_tb_inactive():
            if event.inaxes == self.view.sk_canvas:
                print("Canvas was clicked.")
                event.canvas.release_mouse(self.view.sk_canvas)

                if self.action is Actions.set_min_bound:
                    self.model.crv_bnds[0] = self.view.borders[0].radius
                    print("Inner bound set: {}".format(
                          self.model.crv_bnds[0]))
                    self.action = Actions.set_max_bound
                elif self.action is Actions.set_max_bound:
                    self.model.crv_bnds[1] = self.view.borders[1].radius
                    print("Outer bound set: {}".format(
                          self.model.crv_bnds[1]))
                    self.action = Actions.none
                elif self.action is Actions.sketch:
                    self.action = Actions.none

            elif self.action is Actions.search:
                if event.inaxes == self.view.overlay:
                    self.quit_search_mecha()

            elif event.inaxes in [s.ax for s in
                                  self.view.controls.sliders.values()]:
                self.model.compute_pois()


def main():
    c = Controller()
    c.run()


if __name__ == "__main__":
    main()
