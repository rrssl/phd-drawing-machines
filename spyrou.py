# -*- coding: utf-8 -*-
"""
Spyrou, the drawing-machine designer.

@author: Robin Roussel
"""
from enum import Enum
import math

from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

from controlpane import ControlPane, make_slider
from mecha import EllipticSpirograph
from pois import find_krv_max, find_isect


Modes = Enum('Modes', 'sketcher editor')
Actions = Enum('Actions',
               'none sketch set_min_bound set_max_bound set_sym search show')


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


class Model:

    def __init__(self):
        self.crv_bnds = [None, None]
        self.sym_order = 1
        self.strokes = []
        self.undone_strokes = []
        # List of dicts with keys ('type', 'props', 'curve')
        self.search_res = []
        self.mecha = None
        self.dprops = []
        self.cprops = []
        self.nb_cp = 2
        self.nb_dp = 2
        self.cbounds = []
        self.crv = None
        self.krv_pois = None
        self.isect_pois = None

    def find_pois(self):
        self.krv_pois = find_krv_max(self.crv)
        self.isect_pois = find_isect(self.crv)
#        print(self.krv_pois)
#        print(self.isect_pois)

    def search_mecha(self, nb):
        self.search_res.clear()
        mecha = EllipticSpirograph(5, 3, .2, 1.)
        for _ in range(nb):
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
        self.find_pois()

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

        self.ed_head = None
        self.ed_layer = []
        self.controls = None
        self.draw_editor_tab()
        self.update_controls([], [], redraw=False)
        self.hide_layer(self.ed_layer)

        self.draw_bottom_pane()

        self.overlay = self.draw_overlay()
        self.hide_layer([self.overlay])

        # Wait until first call of show_search_menu to instantiate axes.
        self.se_layer = []

        self.undone_plots = []
        self.borders = [None, None]
        self.sym_lines = []
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

    def draw_button(self, grid_pos, label, width):
        bt_ax = plt.subplot2grid(
            self.grid_size, grid_pos, rowspan=2, colspan=width)
        bt = FancyButton(bt_ax, label, color='.9', hovercolor='lightgreen')
        bt.label.set_fontsize(12)
        bt.label.set_weight('bold')
        bt.label.set_color('.2')
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

        self.sk_head = self.draw_tab_header(
            (0, self.grid_size[0]), "Sketcher", active=True)

        self.sk_layer.append(
            self.draw_section_title((3, self.grid_size[0]+tab_width//2),
                                    "Construction lines")
            )
        self.widgets['sk_bnd'] = self.draw_button(
            (4, self.grid_size[0]+tab_width//2), "Set boundaries",
            tab_width)
        self.sk_layer.append(self.widgets['sk_bnd'].ax)

        slider_args = {'valmin': 0, 'valmax': 25, 'valinit': 1,
                       'label': "Symmetry\norder"}
        self.widgets['sk_sym'] = self.draw_slider(
            (7, self.grid_size[0]+tab_width*3//4), slider_args,
            tab_width*5//4)
        self.sk_layer.append(self.widgets['sk_sym'].ax)

        self.sk_layer.append(
            self.draw_separator((8, self.grid_size[0]))
            )
        self.sk_layer.append(
            self.draw_section_title((10, self.grid_size[0]+tab_width//2),
                                    "Sketch")
            )
        self.widgets['sk_undo'] = self.draw_button(
            (11, self.grid_size[0]+tab_width//4), "Undo\nlast stroke",
            tab_width*3//4)
        self.sk_layer.append(self.widgets['sk_undo'].ax)
        self.widgets['sk_redo'] = self.draw_button(
            (11, self.grid_size[0]+tab_width*5//4), "Redo\nlast stroke",
            tab_width*3//4)
        self.sk_layer.append(self.widgets['sk_redo'].ax)

    def draw_editor_tab(self):
        tab_width = (self.grid_size[1] - self.grid_size[0]) // 2

        # Put material in sketcher tab aside so that there is not conflict
        # when instantiating axes. Yes, it's ugly.
        for panel in self.sk_layer:
            pos = panel.get_position()
            pos.y0 -= 2*tab_width
            pos.y1 -= 2*tab_width
            panel.set_position(pos)

        self.ed_head = self.draw_tab_header(
            (0, self.grid_size[0]+tab_width), "Editor")

        self.ed_layer.append(
            self.draw_section_title((3, self.grid_size[0]+tab_width//2),
                                    "Invariants")
            )

        self.ed_layer.append(
            self.draw_separator((8, self.grid_size[0]))
            )

        self.ed_layer.append(
            self.draw_section_title((10, self.grid_size[0]+tab_width//2),
                                    "Controls")
            )

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
            (11, self.grid_size[0]+tab_width//4), data, bounds, tab_width*7//4)
        for id_, slider in self.controls.sliders.items():
            self.widgets['ed_prop_{}'.format(id_)] = slider
            self.ed_layer.append(slider.ax)

        # Put back the slider tab in place.
        for panel in self.sk_layer:
            pos = panel.get_position()
            pos.y0 += 2*tab_width
            pos.y1 += 2*tab_width
            panel.set_position(pos)

    def draw_bottom_pane(self):
        width = (self.grid_size[1] - self.grid_size[0]) // 2
        self.draw_separator((15, self.grid_size[0]))
        self.draw_section_title((17, self.grid_size[0]+width//2),
                                "Drawing machine")
        self.widgets['search'] = self.draw_button(
            (18, self.grid_size[0]+width//2), "Search database", width)
        self.widgets['show'] = self.draw_button(
            (21, self.grid_size[0]+width//4), "Show\nmechanism", width*3//4)
        self.widgets['export'] = self.draw_button(
            (21, self.grid_size[0]+width*5//4), "Export\nmechanism",
            width*3//4)

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

    def update_pois(self, pois_xy):
        for item in self.ed_canvas.patches:
            if item not in self.locked_pois:
                item.remove()
        if pois_xy.size:
            locked_xy = [poi.center.tolist() for poi in self.locked_pois]
            for xy in pois_xy:
                if xy.tolist() not in locked_xy:
                    self.ed_canvas.add_patch(
                        Circle(xy, radius=.1, fill=True, fc='lightgreen',
                               ec='none', zorder=3, picker=True)
                        )

    def lock_poi(self, poi_patch):
        self.locked_pois.append(poi_patch)
        poi_patch.set_facecolor('none')
        poi_patch.set_edgecolor('lightgreen')
        poi_patch.set_linewidth(4)

    def unlock_poi(self, poi_patch):
        self.locked_pois.remove(poi_patch)
        poi_patch.set_facecolor('lightgreen')
        poi_patch.set_edgecolor('none')
        poi_patch.set_linewidth(1)


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
        if self.model.krv_pois is None:
            if self.model.isect_pois is None:
                return None
            else:
                pois = self.model.isect_pois[:, :2]
        else:
            if self.model.isect_pois is None:
                pois = self.model.krv_pois[:, :2]
            else:
                pois = np.vstack([self.model.krv_pois[:, :2],
                                  self.model.isect_pois[:, :2]])
        position = np.asarray(position).reshape(1, 2)
        extent = self.view.ed_canvas.get_xlim()
        radius = self.poi_capture_radius * abs(extent[1] - extent[0])
        valid = np.linalg.norm(pois-position, axis=1) <= radius
        return pois[valid, :]

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
                pois = self.find_close_pois((event.xdata, event.ydata))
                self.view.update_pois(pois)
                self.view.redraw_axes(self.view.ed_canvas)

    def on_pick(self, event):
        if self.check_tb_inactive():
            if event.mouseevent.inaxes == self.view.ed_canvas:
                if event.artist in self.view.locked_pois:
                    self.view.unlock_poi(event.artist)
                else:
                    self.view.lock_poi(event.artist)
                self.view.redraw_axes(self.view.ed_canvas)

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
                self.model.find_pois()


def main():
    c = Controller()
    c.run()


if __name__ == "__main__":
    main()