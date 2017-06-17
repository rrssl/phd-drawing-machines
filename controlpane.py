# -*- coding: utf-8 -*-
"""
Control pane (with sliders)

@author: Robin Roussel
"""
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.widgets import Slider

from discreteslider import DiscreteSlider


class SliderBound:
    """Used to show dynamic bounds superimposed on the slider."""

    def __init__(self, sliderax, valmin, valmax, type_='lower'):
        self.poly = sliderax.axvspan(valmin, valmax, 0, 1, alpha=.8, color='r')
        # Order of the vertices:
        # 1--2
        # |  |
        # 0--3
        if type_ == 'lower':
            self.val = valmax
            self.xy_ids = (2, 3)    # Poly's vertices to update in set_val.
        elif type_ == 'upper':
            self.val = valmin
            self.xy_ids = (1, 0)
        else:
            print("Unknown type.")

    def set_val(self, val):
        self.val = val
        xy = self.poly.xy
        xy[self.xy_ids[0]] = val, 1
        xy[self.xy_ids[1]] = val, 0
        self.poly.xy = xy


def make_slider(ax, bounds=None, show_init=False, show_value=True, **kwargs):
    """Factory function with more options than the classic Slider.

    See ControlPane doc for info on the arguments.
    """
    # Add optional bounds.
    if bounds is not None:
        low, up = bounds
        if low is not None:
            kwargs['slidermin'] = SliderBound(
                ax, kwargs.get('valmin'), low, type_='lower')
        if up is not None:
            kwargs['slidermax'] = SliderBound(
                ax, up, kwargs.get('valmax'), type_='upper')
    # Create slider.
    if (type(kwargs.get('valmin')) == int and
            type(kwargs.get('valmax')) == int):
        vals = range(kwargs.get('valmin'), kwargs.get('valmax') + 1)
        slider = DiscreteSlider(ax, allowed_vals=vals, **kwargs)
    else:
        slider = Slider(ax, **kwargs)
    if not show_init:
        slider.vline.remove()
    slider.valtext.set_visible(show_value)
    return slider


class ControlPane:
    """Control pane.

    Parameters
    ----------
    figure : Figure
        matplotlib figure.
    param_data : sequence
        sequence of (id, kwargs) pairs, where 'id' is whatever identifier you
        want to give to the callback when the corresp. slider is changed, and
        'kwargs' contains the Slider arguments.
    update_func : callable
        callable object with arguments (id, value).
    subplot_spec : SubplotSpec, optional
        Defines the space in which to draw the control pane; if not specified,
        room will be made at the bottom of the figure.
    bounds : sequence, optional
        sequence of (lower, upper) pairs (same order as param_data), defining
        bounds within each slider (can be changed dynamically).
    show_init : boolean, optional
        Whether or not to mark the initial position with a vertical bar.
    show_value : boolean, optional
        Whether or not to show the current value on the side of the slider.

    If objects of type int are given as 'valmin' and 'valmax', a discrete
    slider will be used.
    """

    def __init__(self, figure, param_data, update_func, subplot_spec=None,
                 bounds=None, show_init=False, show_value=True):
        self.fig = figure
        self.param_data = param_data
        self.update = update_func
        self.subspec = subplot_spec
        self.bounds = bounds
        self.show_init = show_init
        self.show_value = show_value

        self.sliders = {}

        self.draw()

    def _create_update_func(self, id_):
        return lambda val: self.update(id_, val)

    def draw(self):
        """Draw the control pane."""
        # Adjust the layout.
        N = len(self.param_data)
        if self.subspec is None:
            self.fig.subplots_adjust(bottom=.1 + .05*N)
        else:
            gs = GridSpecFromSubplotSpec(N, 1, subplot_spec=self.subspec)

        for i, (id_, args) in enumerate(self.param_data):
            if args.get('alpha') is None:
                args['alpha'] = .5
            # Create Axes instance.
            if self.subspec is None:
                ax = self.fig.add_axes([.3, .05 * (N-i), .4, .02])
            else:
                ax = self.fig.add_subplot(gs[i, :])
            # Create slider.
            bnds = None if self.bounds is None else self.bounds[i]
            slider = make_slider(ax, bnds, self.show_init, self.show_value,
                                 **args)
            if self.update is not None:
                slider.on_changed(self._create_update_func(id_))
            self.sliders[id_] = slider

        self.fig.canvas.draw()

    def set_val(self, id_, val, incognito=False):
        """Change the value of slider 'id_'.
        Incognito mode will not fire callbacks nor redraw the slider.
        """
        s = self.sliders.get(id_)
        if s is None:
            return

        if incognito:
            flags = s.drawon, s.eventson
            s.drawon, s.eventson = False, False

        s.set_val(val)

        if incognito:
            s.drawon, s.eventson = flags

    def set_bounds(self, id_, bounds):
        """Change the dynamic bounds of slider 'id_'."""
        assert(self.bounds is not None)
        self.sliders[id_].slidermin.set_val(bounds[0])
        self.sliders[id_].slidermax.set_val(bounds[1])

    def set_valminmax(self, id_, valmin, valmax):
        """Change the valmin and valmax of slider 'id_'."""
        slider = self.sliders[id_]
        slider.valmin = valmin
        slider.valmax = valmax
        slider.ax.set_xlim((valmin, valmax))
        # Update poly
        xy = slider.poly.xy
        xy[0] = valmin, 0
        xy[1] = valmin, 1
        slider.poly.xy = xy
        # Update bounds' polys
        if slider.slidermin is not None:
            xy_smin = slider.slidermin.poly.xy
            xy_smin[0] = valmin, 0
            xy_smin[1] = valmin, 1
            slider.slidermin.poly.xy = xy_smin
        if slider.slidermax is not None:
            xy_smax = slider.slidermax.poly.xy
            xy_smax[2] = valmax, 1
            xy_smax[3] = valmax, 0
            slider.slidermax.poly.xy = xy_smax
