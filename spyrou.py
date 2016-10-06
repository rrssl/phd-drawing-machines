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

from controlpane import ControlPane
from mecha import EllipticSpirograph

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
        for _, s in ax.spines.items():
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
        self.search_res = []
        self.mecha = None

    def search_mecha(self, nb):
        self.search_res.clear()
        mecha = EllipticSpirograph(5, 3, .2, 1.)
        for _ in range(nb):
            self.search_res.append(mecha.get_curve())


class View:

    def __init__(self, fig_size=(12,8)):
        self.fig = plt.figure(figsize=fig_size, facecolor='.2')
        self.fig.canvas.set_window_title('Spyrou2.0')
        plt.subplots_adjust(left=0., right=1., bottom=0., top=1.,
                            wspace=0., hspace=0.)
        self.grid_size = (3*fig_size[1], 3*fig_size[0])

        self.canvas = self.draw_canvas()
        self.bts = {}
        self.sk_head = None
        self.sk_layer = []
        self.draw_sketcher_tab()
        self.ed_head = None
        self.ed_layer = []
        self.draw_editor_tab()
        self.draw_bottom_pane()

        self.undone_plots = []
        self.borders = [None, None]
        self.sym_lines = []
        self.res_cells = []

    def draw_canvas(self):
        canvas = plt.subplot2grid(
            self.grid_size, (0, 0), rowspan=self.grid_size[0],
            colspan=self.grid_size[0])
        canvas.get_xaxis().set_ticks([])
        canvas.get_yaxis().set_ticks([])
        canvas.set_xlim([-1, 1])
        canvas.set_ylim([-1, 1])
        canvas.set_aspect('equal', 'datalim')
        for _, s in canvas.spines.items():
            s.set_color('none')
#        canvas.spines['bottom'].set_color('none')
#        canvas.spines['top'].set_color('none')
#        canvas.spines['right'].set_color('none')
#        canvas.spines['left'].set_color('none')
        return canvas

    def draw_sketcher_tab(self):
        tab_width = (self.grid_size[1] - self.grid_size[0]) // 2

        self.sk_head = self.draw_tab_header(
            (0, self.grid_size[0]), "Sketcher", active=True)

        self.sk_layer.append(
            self.draw_section_title((3, self.grid_size[0]+tab_width//2),
                                    "Construction lines")
            )

        self.bts['sk_bnd'] = self.draw_button(
            (4, self.grid_size[0]+tab_width//2), "Set boundaries",
            tab_width)
        self.sk_layer.append(self.bts['sk_bnd'].ax)

#        self.bts['sk_sym'] = self.draw_button(
#            (6, self.grid_size[0]+tab_width*5//4), "Set\nsymmetry",
#            tab_width*3//4)
        self.sym_slider = self.draw_sym_slider(
            (7, self.grid_size[0]+tab_width*3//4), "Symmetry\norder",
            tab_width*5//4)
        self.sk_layer.append(self.sym_slider.ax)

        self.sk_layer.append(
            self.draw_separator((8, self.grid_size[0]))
            )
        self.sk_layer.append(
            self.draw_section_title((10, self.grid_size[0]+tab_width//2),
                                    "Sketch")
            )
        self.bts['sk_undo'] = self.draw_button(
            (11, self.grid_size[0]+tab_width//4), "Undo\nlast stroke",
            tab_width*3//4)
        self.sk_layer.append(self.bts['sk_undo'].ax)
        self.bts['sk_redo'] = self.draw_button(
            (11, self.grid_size[0]+tab_width*5//4), "Redo\nlast stroke",
            tab_width*3//4)
        self.sk_layer.append(self.bts['sk_redo'].ax)

    def draw_editor_tab(self):
        tab_width = (self.grid_size[1] - self.grid_size[0]) // 2
        self.ed_head = self.draw_tab_header(
            (0, self.grid_size[0]+tab_width), "Editor")

    def draw_bottom_pane(self):
        width = (self.grid_size[1] - self.grid_size[0]) // 2
        self.draw_separator((14, self.grid_size[0]))
        self.draw_section_title((16, self.grid_size[0]+width//2),
                                "Drawing machine")
        self.bts['search'] = self.draw_button(
            (18, self.grid_size[0]+width//2), "Search database", width)
        self.bts['show'] = self.draw_button(
            (21, self.grid_size[0]+width//4), "Show\nmechanism", width*3//4)
        self.bts['export'] = self.draw_button(
            (21, self.grid_size[0]+width*5//4), "Export\nmechanism", width*3//4)

    def draw_tab_header(self, grid_pos, label, active=False):
        width = (self.grid_size[1] - self.grid_size[0]) // 2
        tab_ax = plt.subplot2grid(
            self.grid_size, grid_pos, rowspan=2, colspan=width)
        for _, s in tab_ax.spines.items():
            s.set_color('none')
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
            self.grid_size, grid_pos, rowspan=1, colspan=width)
        for _, s in ax.spines.items():
            s.set_color('none')
        ax.set_navigate(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_axis_bgcolor('none')
        title = ax.text(0.5, 0.5, label,
                               verticalalignment='bottom',
                               horizontalalignment='center',
                               transform=ax.transAxes)
        title.set_fontsize(15)
        title.set_weight('light')
        title.set_color('.9')
        return ax

    def draw_sym_slider(self, grid_pos, label, width):
        gs = GridSpec(*self.grid_size)
        sub_spec = gs[grid_pos[0]:grid_pos[0]+1,
                      grid_pos[1]:grid_pos[1]+width]
#        ax = plt.subplot2grid(
#            self.grid_size, grid_pos, rowspan=1, colspan=width)
        data = (
            ('sym', {'valmin': 1,
                     'valmax': 25,
                     'valinit': 1,
                     'label': label,
                     'color': 'lightgreen'
                    }),
            )
        cp = ControlPane(self.fig, data, None, sub_spec, show_value=True)
        slider = cp.sliders['sym']
        slider.ax.set_axis_bgcolor('.9')
        for _, s in slider.ax.spines.items():
            s.set_color('none')
        slider.poly.set_alpha(1)
        slider.label.set_weight('bold')
        slider.label.set_color('.9')
        slider.label.set_x(-0.25)
        slider.label.set_horizontalalignment('center')
        slider.valtext.set_weight('bold')
        slider.valtext.set_color('.9')
        slider.disconnect(0)
        return cp.sliders['sym']

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
            self.grid_size, grid_pos, rowspan=1, colspan=width)
        for _, s in ax.spines.items():
            s.set_color('none')
        ax.set_navigate(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_axis_bgcolor('none')
        ax.axhline(.1, 0.01, .99, color='.5', linestyle='dashed')
        return ax

    def draw_overlay(self):
        ol = self.fig.add_axes([0, 0, 1, 1])
        ol.set_navigate(False)
        ol.get_xaxis().set_ticks([])
        ol.get_yaxis().set_ticks([])
        for _, s in ol.spines.items():
            s.set_color('none')
        ol.patch.set_alpha(.5)
        self.overlay = ol

    def draw_inner_bound(self, radius):
        self.borders[0] = Circle((0, 0), radius, fc='.9', ec='.7', ls='dashed')
        self.canvas.add_patch(self.borders[0])

    def draw_outer_bound(self, radius):
        self.canvas.set_axis_bgcolor('.9')
        self.borders[1] = Circle(
            (0, 0), radius, fc='1.', ec='.7', ls='dashed', zorder=-1)
        self.canvas.add_patch(self.borders[1])

    def draw_search_result_cell(self, coords, drawing):
        ax = self.fig.add_axes(coords, axisbg='.9', zorder=2)
        ax.set_aspect('equal', 'datalim')
        ax.margins(.1)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_navigate(False)
        for _, s in ax.spines.items():
            s.set_color('.2')
            s.set_linewidth(10)
        ax.plot(*drawing)
        return ax

    def draw_search_results_pane(self, results, nb_col=3):
        nb_res = len(results)
        nb_rows = math.ceil(nb_res / nb_col)
        cell_size = .15
        ratio = self.grid_size[1] / self.grid_size[0]
        margin = cell_size * .1

        x0 = .5 - cell_size*nb_col/2
        y0 = .5 + cell_size*ratio*(nb_rows/2 - 1)
        for i, result in enumerate(results):
            x = x0 + (i % nb_col)*cell_size
            y = y0 - (i // nb_col)*cell_size*ratio
            self.res_cells.append(
                self.draw_search_result_cell(
                    [x, y, cell_size, cell_size*ratio], result))

        ax = self.fig.add_axes(
            [x0-margin, y-margin*ratio,
             nb_col*cell_size + 2*margin,
             (nb_rows*cell_size + 2*margin)*ratio],
            axisbg='.2')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_navigate(False)
        for _, s in ax.spines.items():
            s.set_color('none')
        self.search_pane_bg = ax

    def draw_sym_lines(self, order):
        if order > 1:
            radius = 2 * math.sqrt(
                max(np.abs(self.canvas.get_xlim()))**2 +
                max(np.abs(self.canvas.get_ylim()))**2
                )
            for i in range(order):
                angle = 2 * math.pi * i / order
                self.sym_lines.append(
                    self.canvas.plot(
                        (0., radius*math.cos(angle)),
                        (0., radius*math.sin(angle)),
                        c='.7', ls='dashed')[0]
                    )
            # Move these plots at the beginning so that they don't disturb the
            # undo/redo.
            self.canvas.lines = (self.canvas.lines[-order:] +
                                 self.canvas.lines[:-order])

    def redraw_canvas(self):
        self.canvas.redraw_in_frame()
#        self.fig.canvas.blit(self.canvas)
        self.fig.canvas.update()

    def remove_overlay(self):
        self.fig.delaxes(self.overlay)
        self.overlay = None

    def remove_borders(self):
        self.canvas.set_axis_bgcolor('1.')
        for patch in self.borders:
            patch.remove()
        self.borders = [None, None]

    def remove_sym_lines(self):
        for line in self.sym_lines:
            line.remove()
        self.sym_lines = []

    def remove_search_results_pane(self):
        for cell in self.res_cells:
            self.fig.delaxes(cell)
        self.fig.delaxes(self.search_pane_bg)

    def switch_tab(self, mode):
        if mode is Modes.editor:
            self.sk_head.color='.4'
            self.sk_head.hovercolor='.4'
            self.ed_head.color='.2'
            self.ed_head.hovercolor='.2'
            for panel in self.sk_layer:
                panel.set_visible(False)
                panel.set_zorder(-1)
            for name, button in self.bts.items():
                if name.startswith("sk_"):
                    button.set_active(False)
            self.sym_slider.set_active(False)

        elif mode is Modes.sketcher:
            self.sk_head.color='.2'
            self.sk_head.hovercolor='.2'
            self.ed_head.color='.4'
            self.ed_head.hovercolor='.4'
            for panel in self.sk_layer:
                panel.set_visible(True)
                panel.set_zorder(0)
            for name, button in self.bts.items():
                if name.startswith("sk_"):
                    button.set_active(True)
            self.sym_slider.set_active(True)

        self.sk_head.ax.set_axis_bgcolor(self.sk_head.color)
        self.ed_head.ax.set_axis_bgcolor(self.ed_head.color)
        self.fig.canvas.draw()


class Controller:

    def __init__(self):
        self.model = Model()
        self.view = View()
        self.connect_panel()
        self.connect_canvas()

        # Base state
        self.mode = Modes.sketcher
        self.action = Actions.none

    def run(self):
        plt.ioff()
        plt.show()

    def connect_panel(self):
        self.view.sk_head.on_clicked(self.select_sk_tab)
        self.view.ed_head.on_clicked(self.select_ed_tab)
        self.view.bts['sk_bnd'].on_clicked(self.set_bounds)
        self.view.bts['sk_undo'].on_clicked(self.undo_stroke)
        self.view.bts['sk_redo'].on_clicked(self.redo_stroke)
        self.view.bts['search'].on_clicked(self.search_mecha)
        self.view.bts['show'].on_clicked(self.show_mecha)
        self.view.bts['export'].on_clicked(self.export_mecha)
        self.view.sym_slider.on_changed(self.set_symmetry)

    def connect_canvas(self):
        self.view.fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.view.fig.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.view.fig.canvas.mpl_connect(
            'motion_notify_event', self.on_move)

    def check_tb_inactive(self):
        """Check if the matplotlib toolbar plugin is inactive."""
        return self.view.fig.canvas.manager.toolbar._active is None

    def select_sk_tab(self, event):
        if self.mode is Modes.sketcher:
            print("Sketcher mode is already selected.")
        else:
            print("Switching to sketcher mode.")
            self.mode = Modes.sketcher
            self.view.switch_tab(self.mode)

    def select_ed_tab(self, event):
        if self.mode == Modes.editor:
            print("Editor mode is already selected.")
        else:
            print("Switching to editor mode.")
            self.mode = Modes.editor
            self.view.switch_tab(self.mode)

    def set_bounds(self, event):
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
            self.view.undone_plots.append(self.view.canvas.lines.pop())
            self.view.redraw_canvas()

    def redo_stroke(self, event):
        if not len(self.model.undone_strokes):
            print("There is no undone stroke.")
        else:
            print("Redo the last undone sketch stroke.")
            self.model.strokes.append(self.model.undone_strokes.pop())
            self.view.canvas.lines.append(self.view.undone_plots.pop())
            self.view.redraw_canvas()

    def search_mecha(self, event):
        if not len(self.model.strokes):
            print("There is no query sketch.")
        else:
            print("Search for the best matching mechanism.")
            self.action = Actions.search
            self.view.draw_overlay()
            self.model.search_mecha(6)
            self.view.draw_search_results_pane(self.model.search_res)
            self.view.fig.canvas.draw()

    def quit_search_mecha(self):
        print("Quit the drawing selection.")
        self.action = Actions.none
        self.view.remove_overlay()
        self.view.remove_search_results_pane()
        self.view.fig.canvas.draw()

    def show_mecha(self, event):
        if self.model.mecha is None:
            print("There is no mechanism to show.")
        else:
            print("Show the mechanism.")
#            self.active_bt = self.view.bts['show']

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
                self.view.canvas.plot([], [], lw=2, c='b', alpha=.8)
            # Flush history
            self.model.undone_strokes.clear()
            self.view.undone_plots.clear()
        else:
            for i in range(nb_sym):
                angle = 2 * math.pi * i / nb_sym
                cos_, sin_ = math.cos(angle), math.sin(angle)
                xy_ = [xy[0]*cos_ + xy[1]*sin_, -xy[0]*sin_ + xy[1]*cos_]
                self.model.strokes[-nb_sym+i].append(xy_)
                self.view.canvas.lines[-nb_sym+i].set_data(
                    *np.asarray(self.model.strokes[-nb_sym+i]).T)
        self.view.redraw_canvas()

    def set_bound(self, radius):
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
        self.view.redraw_canvas()

    def on_move(self, event):
        if self.check_tb_inactive() and event.inaxes == self.view.canvas:
            if self.mode is Modes.sketcher:
                if (self.action is Actions.set_min_bound
                    or self.action is Actions.set_max_bound):
                    radius = math.sqrt(event.xdata**2 + event.ydata**2)
                    self.set_bound(radius)
                elif self.action is Actions.sketch:
                    self.add_sketch_point([event.xdata, event.ydata])

    def on_press(self, event):
        if self.check_tb_inactive() and event.inaxes == self.view.canvas:
            event.canvas.grab_mouse(self.view.canvas)

            if self.mode is Modes.sketcher:
                if (self.action is Actions.set_min_bound
                    or self.action is Actions.set_max_bound):
                    pass
                else:
                    # Default action is sketching.
                    self.action = Actions.sketch
                    self.add_sketch_point([event.xdata, event.ydata],
                                          start=True)
        elif self.action is Actions.search:
            if event.inaxes == self.view.overlay:
                self.quit_search_mecha()

    def on_release(self, event):
        if self.check_tb_inactive() and event.inaxes == self.view.canvas:
            print("Canvas was clicked.")
            event.canvas.release_mouse(self.view.canvas)

            if self.mode is Modes.sketcher:
                if self.action is Actions.set_min_bound:
                    self.model.crv_bnds[0] = self.view.borders[0].radius
                    print("Inner bound set: {}".format(self.model.crv_bnds[0]))
                    self.action = Actions.set_max_bound
                elif self.action is Actions.set_max_bound:
                    self.model.crv_bnds[1] = self.view.borders[1].radius
                    print("Outer bound set: {}".format(self.model.crv_bnds[1]))
                    self.action = Actions.none
                elif self.action is Actions.sketch:
                    self.action = Actions.none


def main():
    c = Controller()
    c.run()


if __name__ == "__main__":
    main()
