"""
Demo for sketch-based curve retrieval.

"""
import math
import sys

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from joblib import Parallel, delayed
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QSlider, QSpinBox,
                             QVBoxLayout, QWidget)
from scipy.interpolate import splev, splprep
from scipy.signal import fftconvolve
from scipy.spatial import procrustes

import _context  # noqa
import mecha
from curvedistances import DistanceField
from sketcher import Sketcher
from tsp import TSPSolver
#  from curveplotlib import distshow

TYPES = (
    mecha.EllipticSpirograph,
    mecha.SingleGearFixedFulcrumCDM,
    mecha.HootNanny,
)


def get_global_sampling(mecha_type, pts_per_dim, fixed_values=None):
    """Sample feasible parameters for this mechanism."""
    cs = mecha_type.ConstraintSolver
    size = [pts_per_dim] * cs.nb_cprops
    samples = np.array(list(cs.sample_feasible_domain(
        grid_resol=size, fixed_values=fixed_values)))
    return samples


def pad_same_shape(a, b, **pad_kwargs):
    """Pad a and b to the left and bottom, so that they have the same shape."""
    shape_diff = np.subtract(a.shape, b.shape, dtype=int)
    a_pad = np.zeros((shape_diff.size, 2), dtype=int)
    a_pad[:, 1] = -shape_diff.clip(max=0)
    b_pad = np.zeros_like(a_pad)
    b_pad[:, 1] = shape_diff.clip(min=0)
    if a_pad.any():
        a = np.pad(a, a_pad, **pad_kwargs)
    if b_pad.any():
        b = np.pad(b, b_pad, **pad_kwargs)
    return a, b


def get_smooth_drawing(strokes, s=.01):
    if len(strokes) == 0:
        return None
    elif len(strokes) == 1:
        sorted_pts = np.asarray(strokes[0]).T
    else:
        ends = np.vstack([[stroke[0], stroke[-1]] for stroke in strokes])
        n = ends.shape[0]
        distances = {(i, j): ((ends[i] - ends[j]) ** 2).sum()
                     for i in range(2, n) for j in range(i - 1)}
        distances.update({(i - 1, i): ((ends[i - 1] - ends[i]) ** 2).sum()
                          for i in range(2, n, 2)})
        distances.update({(i - 1, i): -1 for i in range(1, n, 2)})
        solver = TSPSolver(n, distances)
        vertices, cost = solver.solve()
        sorted_pts = []
        for i in range(0, n, 2):
            ia, ib = vertices[i], vertices[i + 1]
            if ia < ib:
                sorted_pts.extend(strokes[ia // 2])
            else:
                sorted_pts.extend(strokes[ib // 2][::-1])
        sorted_pts = np.array(sorted_pts).T
    spline = splprep(sorted_pts, s=s, per=1, quiet=1)[0]
    return spline


def get_distances(spline, mecha_type, samples, crv_density):
    mecha = mecha_type(*samples[0])
    #  distance = DistanceField().get_dist()
    values = np.empty(len(samples))
    for i, sample in enumerate(samples):
        mecha.reset(*sample)
        crv = mecha.get_curve(crv_density)
        target = np.array(splev(np.linspace(0, 1, crv.shape[1]), spline))
        #  values[i] = max(distance(crv, target), distance(target, crv))
        values[i] = min(procrustes(target.T, crv.T)[2],
                        procrustes(target.T[::-1], crv.T)[2])
    return values


def lcm(n, m):
    return n * m // math.gcd(n, m)


def find_all_props(spline, mecha_type, disc_props, crv_density):
    disc_props = list(disc_props)
    cs = mecha_type.ConstraintSolver
    # Initial estimate
    print("Initializing optimization-based search")
    samples = cs.sample_feasible_continuous_domain(
        *disc_props, grid_resol=(4,) * cs.nb_cprops)
    samples = list(samples)
    distances = get_distances(spline, mecha_type, samples, crv_density)
    init = samples[distances.argmin()]
    #  return init
    mecha = mecha_type(*init)

    def adapt(cstr):
        return lambda p: cstr(np.concatenate([disc_props, p]))
    cstrs = cs.get_constraints()
    # Start at 2*n_disc_prop to remove constraints on discrete props.
    cstrs = [adapt(cstrs[i]) for i in range(2 * cs.nb_dprops, len(cstrs))]
    #  distance = DistanceField().get_dist
    #  target = np.array(splev(np.linspace(0, 1, 2**8), spline))

    def objective(p):
        mecha.reset(*disc_props + list(p))
        crv = mecha.get_curve(crv_density)
        target = np.array(splev(np.linspace(0, 1, crv.shape[1]), spline))
        dist = min(procrustes(target.T, crv.T)[2],
                   procrustes(target.T[::-1], crv.T)[2])
        return dist
        #  return distance(crv, target)

    print("Optimizing...")
    valid = False
    optinit = init[cs.nb_dprops:]
    niter = 0
    while not valid and niter < 10:
        sol = opt.fmin_cobyla(objective, optinit, cons=cstrs)
        valid = mecha.reset(*disc_props + list(sol))
        if not valid:
            optinit = sol
            niter += 1
    if not valid:
        return init
    else:
        return disc_props + list(sol)


def isinteger(x):
    return np.equal(np.mod(x, 1), 0)


def ismultiple(x, n):
    return isinteger(x / n)


class DrawingFinder:

    def __init__(self):
        # Sketcher
        self.crv_bnds = [None, None]
        self.sym_order = 1
        self.strokes = []  # List of N*2 lists of points
        self.undone_strokes = []
        # Mechanism retrieval
        self.pts_per_dim = {t: 4 for t in TYPES}
        self.distance = DistanceField().get_dist
        self.search_res = []  # list of  {'type', 'props', 'curve'} dicts
        # Mechanism
        self.mecha = None
        self.crv_density = 2**8
        self.crv = None

    def compute_search_space(self, n_samples=2**10):
        search_space = []
        if self.sym_order > 1:
            R = self.sym_order
        else:
            R = None
        if mecha.EllipticSpirograph in TYPES:
            search_space.append(
                    (mecha.EllipticSpirograph, [R, None, None, None]))
        # Compute points
        spline = get_smooth_drawing(self.strokes, s=.01)
        x, y = splev(np.linspace(0, 1, n_samples), spline)
        # Compute Fourier coeffs
        fft = np.fft.fft(x + 1j * y)
        freqs = np.fft.fftfreq(n_samples, 1 / n_samples)
        amp = abs(fft) / n_samples
        amp[0] = 0.  # Center the drawing
        freqs = np.fft.fftshift(freqs)  # Swap the half spaces
        amp = np.fft.fftshift(amp)
        # Find f1
        autocorr = fftconvolve(amp, amp, mode='full')
        f1 = (np.argmax(autocorr) - n_samples + 1) / 2
        if not isinteger(f1):
            # Only known case so far is: r1 = r2
            f1 = freqs[amp.argmax()]
        print("f1 = ", f1)
        # Remove f1
        amp[np.nonzero(freqs == f1)[0][0]] = 0
        freqs -= f1  # Center the peaks
        f1 = int(abs(f1))
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
        f2 = int(freqs[0])
        print("f2 - f1 = ", f2)
        f2m = np.logical_or(ismultiple(freqs, f2), ismultiple(f2, freqs))
        if f2m.all():
            f3 = None  # f3 might be a multiple of f2
        else:
            f3_id = f2m.argmin()
            print(amp, f3_id)
            if amp[f3_id] > amp[0] / 100:
                f3 = int(freqs[f3_id])
            else:
                print("f3 not high enough")
                f3 = None
        print("f3 - f1 = ", f3)
        # Determine the parameters
        if mecha.SingleGearFixedFulcrumCDM in TYPES:
            if R is not None:
                r1 = R * f1 / f2
            else:
                r1 = None
            search_space.append(
                    (mecha.SingleGearFixedFulcrumCDM,
                     [R, r1, None, None, None, None]))
        if mecha.HootNanny in TYPES:
            if f3 is not None:
                period = lcm(int(f1), lcm(int(f2), int(f3)))
                R_ = period // f1
                r1 = period // f2
                r2 = period // f3
            else:
                if R is not None:
                    period = R * f1
                    R_ = R
                    r1 = period // f2
                    r2 = 1
                else:
                    R_ = f2
                    r1 = f1
                    r2 = None
            search_space.append(
                    (mecha.HootNanny,
                     [R_, r1, r2, None, None, None, None, None]))
            search_space.append(
                    (mecha.HootNanny,
                     [R_, r2, r1, None, None, None, None, None]))
        return search_space

    def search_mecha(self, n_res):
        """Retrieve the closest drawings and their associated mechanisms."""
        if not len(self.strokes):
            return
        self.search_res.clear()
        spline = get_smooth_drawing(self.strokes)
        self.spline = spline
        # Sample the parameter space.
        search_space = self.compute_search_space()
        print("Search space:")
        print(search_space)
        samples = Parallel(n_jobs=len(search_space))(
                delayed(get_global_sampling)(t, self.pts_per_dim[t], fv)
                for t, fv in search_space)
        print("Num samples", sum(len(s) for s in samples))
        samples = [(ss[0], s) for ss, s in zip(search_space, samples)]
        # Compute distances.
        distances = Parallel(n_jobs=len(search_space))(
                delayed(get_distances)(spline, t, s, self.crv_density)
                for t, s in samples)
        distances = np.concatenate(distances)
        best = distances.argpartition(n_res)[:n_res]
        # Sort the best matching curves.parameter
        best = best[distances[best].argsort()]
        best_dist = distances[best]
        print("Best distances", best_dist)
        # Build the result.
        ranges = np.cumsum([len(s[1]) for s in samples])
        type_ids = np.digitize(best, ranges)
        ranges = np.insert(ranges, 0, 0)
        best -= ranges[type_ids]
        for bid, tid, dist in zip(best, type_ids, best_dist):
            mecha_type = samples[tid][0]
            print(samples[tid][1][bid])
            mecha = mecha_type(*samples[tid][1][bid])
            self.search_res.append({
                'type': mecha_type,
                'props': mecha.props.copy(),
                'curve': mecha.get_curve(self.crv_density),
                'dist': dist
            })
        return self.search_res


def HLine():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


class Window(QWidget):

    def __init__(self):
        super().__init__()

        self.finder = DrawingFinder()
        self.initUI()

    def initUI(self):
        # Define matplotlib context for the sketcher.
        fig = Figure(figsize=(6, 6), dpi=100)
        self.fig = fig
        can = FigureCanvas(fig)
        can.setParent(self)
        can.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        can.updateGeometry()
        # Add the sketcher.
        fig.subplots_adjust(left=0., right=1., bottom=0., top=1.,
                            wspace=0., hspace=0.)
        ax = fig.add_subplot(111)
        self.sketcher = Sketcher(ax, self.finder)
        # Add buttons.
        #  bnd_bt = QPushButton("Set bounds", self)
        undo_bt = QPushButton("Undo stroke", self)
        redo_bt = QPushButton("Redo stroke", self)
        search_bt = QPushButton("Search drawing", self)
        show_bt = QPushButton("Show spline", self)
        save_bt = QPushButton("Save canvas", self)
        # Add symmetry slider.
        sym_sld = QSlider(Qt.Horizontal, self)
        sym_sld.setTickInterval(1)
        sym_sld.setTickPosition(QSlider.TicksBelow)
        sym_sld.setToolTip("Change the order of rotational symmetry")
        sym_sld.setValue(1)
        sym_sld.setRange(1, 10)
        sym_sld.setFocusPolicy(Qt.NoFocus)
        # Add text box.
        sym_txt = QLabel("Symmetry order: 1", self)
        sym_txt.setAlignment(Qt.AlignCenter)
        self.sym_txt = sym_txt
        # Connect callbacks.
        #  bnd_bt.clicked.connect(self.sketcher.set_sketch_bounds)
        undo_bt.clicked.connect(self.sketcher.undo_stroke)
        redo_bt.clicked.connect(self.sketcher.redo_stroke)
        search_bt.clicked.connect(self.search_mecha)
        show_bt.clicked.connect(self.show_spline)
        save_bt.clicked.connect(self.save_canvas)
        sym_sld.valueChanged.connect(self.set_symmetry)
        # Arrange widgets.
        top = QHBoxLayout()
        top.addWidget(can)
        panel = QVBoxLayout()
        top.addLayout(panel)
        #  panel.addSpacing(10)
        #  panel.addWidget(bnd_bt)
        panel.addSpacing(10)
        panel.addWidget(sym_txt)
        panel.addWidget(sym_sld)
        panel.addSpacing(10)
        undoredo = QHBoxLayout()
        panel.addLayout(undoredo)
        undoredo.addWidget(undo_bt)
        undoredo.addWidget(redo_bt)
        panel.addSpacing(10)
        panel.addWidget(HLine())
        panel.addWidget(search_bt)
        panel.addWidget(HLine())
        panel.addStretch()
        panel.addWidget(show_bt)
        panel.addWidget(save_bt)
        self.setLayout(top)
        # Finalize window
        self.setGeometry(10, 10, 800, 600)  # left, top, width, height
        self.setWindowTitle("Drawing finder")

    def set_symmetry(self, value):
        self.sketcher.set_symmetry(value)
        self.sym_txt.setText("Symmetry order: {}".format(value))

    def search_mecha(self, event):
        if not len(self.finder.strokes):
            print("There is no query sketch.")
        else:
            print("Search for the best matching mechanism.")
            n = 9
            results = self.finder.search_mecha(n)

            spline = self.finder.spline
            dims = (n // 3 + bool(n % 3), 3)
            fig, axes = plt.subplots(
                dims[0], dims[1], figsize=plt.figaspect(1 / 2),
                subplot_kw=dict(aspect='equal'))
            for ax, res in zip(axes.flat, results):
                #  distshow(ax, sketch, res['curve'])
                curve = res['curve'].T
                sketch = np.column_stack(
                    splev(np.linspace(0, 1, len(curve)), spline))
                curve, sketch, _ = procrustes(curve, sketch)
                ax.plot(*curve.T)
                #  ax.plot(*sketch.T)
                props = np.round(res['props'], 2)
                ax.set_title("Distance: {:.2f}\nParameters: {}".format(
                    res['dist'], props))
                ax.axis("off")
            #  fig.tight_layout()
            fig.show()

    def show_spline(self, event):
        strokes = self.finder.strokes
        spline = get_smooth_drawing(strokes, s=.01)
        n_samples = 2**10
        x, y = np.array(splev(np.linspace(0, 1, n_samples), spline))
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.axis('off')
        for stroke in strokes:
            stroke = np.array(stroke).T
            ax.plot(*stroke, c='tab:blue')
        ax.plot(x, y, c='tab:orange')
        fig.tight_layout()
        fig.show()

        #  if 1:
        #      #  self.sketcher.canvas.plot(*points, c='red')
        #      self.sketcher.canvas.scatter(
        #          *points, c=np.linspace(0, 1, points.shape[1]), alpha=.5)
        #      self.sketcher.redraw_axes()
        #      #  fft = np.fft.fft(points[0] + 1j * points[1])
        #      #  mask = np.ones(fft.shape, dtype=bool)
        #      #  mask[modes.astype(int)] = False
        #      #  fft[mask] = 0
        #      #  points2 = np.fft.ifft(fft)
        #      #  self.sketcher.canvas.plot(
        #      #          points2.real, points2.imag, c='orange')
        #      self.sketcher.redraw_axes()

    def save_canvas(self):
        self.fig.savefig("canvas.svg")


def main():
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
