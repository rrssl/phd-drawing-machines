"""
Base module for the forward controller app.

@author: Robin Roussel
"""
import json
import math
import random
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
import numpy as np
from scipy.signal import fftconvolve

import _context
from mechaplot import mechaplot_factory
from controlpane import ControlPane


def isinteger(x):
    return np.equal(np.mod(x, 1), 0)


def ismultiple(x, y):
    return isinteger(x / y)


def lcm(n, m):
    return n * m // math.gcd(n, m)


class ForwardController:
    """Base class for the forward controller app.

    Parameters
    ----------
    mecha_type: Mechanism
        As defined in the mecha module.
    param_data: iterable
        Sequence of (param_id, dict) pairs, where each dict defines
        the parameters required to initialize the sliders.
    """

    def __init__(self, mecha_type, param_data, pt_density=2**6):
        self.param_data = param_data
        self.pt_density = pt_density

        self.mecha = mecha_type(
            *[d.get('valinit') for _, d in param_data])
        self.crv = self.mecha.get_curve(nb=self.pt_density)

        self._init_draw(param_data)

    def _init_draw(self, param_data):
        self.fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(9, 12)
        self.ax = self.fig.add_subplot(gs[:, :6])
        self.ax.set_aspect('equal')
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        plt.subplots_adjust(left=.05, wspace=0., hspace=1.)

        self.crv_plot = self.ax.plot([], [], lw=1, alpha=.8)[0]
        # Since the paper may rotate with the turntable, we pass the drawing.
        self.mecha_plot = mechaplot_factory(self.mecha, self.ax, self.crv_plot)

        bounds = [self.get_bounds(i) for i in range(len(self.mecha.props))]
        self.control_pane = ControlPane(
            self.fig, param_data, self.update,
            subplot_spec=gs[:-2, 8:], bounds=bounds)

        btn_ax = self.fig.add_subplot(gs[-1, 7:9])
        self.gen_btn = Button(btn_ax, "Generate random\ncombination")
        self.gen_btn.on_clicked(self.generate_random_params)

        btn_ax = self.fig.add_subplot(gs[-1, 10:])
        self.sv_btn = Button(btn_ax, "Save combination")
        self.sv_btn.on_clicked(self.save_params)

        btn_ax = self.fig.add_subplot(gs[-2, 10:])
        self.info_btn = Button(btn_ax, "Show info")
        self.info_btn.on_clicked(self.show_info)

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
                    params[i] = random.random() * (b - a) + a
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
            with open("../saved_params.json", "r") as file:
                data = json.load(file)
                data.append(save)
        except FileNotFoundError:
                data = [save]
        with open("../saved_params.json", "w") as file:
                json.dump(data, file)
        print('Successfully saved {}'.format(save))

    def show_info(self, event):
        x, y = self.crv
        #  x = x[:-1]
        #  y = y[:-1]

        n_samples = len(x)
        frequency = 1 / n_samples
        points = x + 1j * y
        print("Samples: ", n_samples)

        fft = np.fft.fft(points)
        freqs = np.fft.fftfreq(n_samples, frequency)
        amp = abs(fft) / n_samples
        phases = np.angle(fft)

        freqs = np.fft.fftshift(freqs)
        amp = np.fft.fftshift(amp)
        phases = np.fft.fftshift(phases)

        #  autocorr = np.correlate(amp, amp[::-1], mode='full')
        autocorr = fftconvolve(amp, amp, mode='full')
        shifts = (np.arange(autocorr.size) - n_samples + 1) / 2

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.plot(freqs, amp)
        ax1.set_title("Fourier coefficients")
        ax2.plot(shifts, autocorr)
        ax2.set_title("Autocorrelation of the Fourier coefficients")
        plt.show(block=False)

        # Find f1
        amp[0] = 0
        autocorr = fftconvolve(amp, amp, mode='full')
        f1 = (np.argmax(autocorr) - n_samples + 1) / 2
        if not isinteger(f1):
            # Only known case so far is: r1 = r2
            f1 = freqs[amp.argmax()]
        print("f1 = ", f1)
        # Remove f1
        amp[np.nonzero(freqs == f1)[0][0]] = 0
        freqs -= f1  # Center the peaks
        f1 = abs(f1)
        # Keep the side with the highest peaks
        posf = freqs >= 0
        negf = freqs <= 0
        if amp[posf].sum() >= amp[negf].sum():
            freqs = freqs[posf]
            amp = amp[posf]
        else:
            freqs = -freqs[negf][::-1]
            amp = amp[negf][::-1]
        # Extract a conservative number of peaks
        n_peaks = 2
        modes_ids = np.argsort(amp)[-n_peaks:][::-1]
        # Process the peaks
        freqs = freqs[modes_ids]
        print(freqs)
        amp = amp[modes_ids]
        print(amp)
        f2 = freqs[0]
        print("f2 - f1 = ", f2)
        f2m = np.logical_or(ismultiple(freqs, f2), ismultiple(f2, freqs))
        if f2m.all():
            f3 = None  # f3 might be a multiple of f2
        else:
            f3_id = f2m.argmin()
            print(amp, f3_id)
            if amp[f3_id] > amp[0] / 10:
                f3 = freqs[f3_id]
            else:
                f3 = None
        print("f3 - f1 = ", f3)
        if f3 is not None:
            period = lcm(int(f1), lcm(int(f2), int(f3)))
            R = period / f1
            r1 = period / f2
            r2 = period / f3
            print("Parameters: ", R, r1, r2)
        #  n_modes_sym = 5
        #  modes_ids = np.sort(np.argpartition(amp, -n_modes_sym)[-n_modes_sym:])
        #  freqs = freqs[modes_ids]
        #  amp = amp[modes_ids]
        #  phases = phases[modes_ids]
        #  print("")
        #  print(freqs)
        #  print(np.round(amp, 3))
        #  print(np.round(phases, 3))
        #  print(np.angle(fft[~mask]))
        #  points = np.fft.ifft(fft)
        #  fig, ax = plt.subplots()
        #  ax.plot(points.real, points.imag)
        #  ax.set_aspect('equal')
        #  plt.show()

    def get_bounds(self, i):
        a, b = self.mecha.get_prop_bounds(i)
        if (i >= self.mecha.ConstraintSolver.nb_dprops and
                np.isfinite((a, b)).all()):
            # Account for slider imprecision wrt bounds.
            margin = (b - a) / 1000
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
