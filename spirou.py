#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application for the retrieval and editing of Spirograph curves.

@author: Robin Roussel
"""
import matplotlib.patches as mpat
import matplotlib.pyplot as plt
import numpy as np

import artist as art
import curvedistances as cdist
import curvegen as cg
import curvematching as cmat

class Spirou():
    """Main application."""

    def __init__(self, init_curve_type, matcher=None, optimizer=None):
        self.curve_type = init_curve_type

        self.samples = init_curve_type.sample_parameters((8,8,4,4))
#        self.samples = init_curve_type.sample_parameters((10,))
        self.matcher = matcher
        self.optimizer = optimizer
        self.update_matcher()

        self.nb_retrieved = 6
        self.init_draw()

    def init_draw(self):
        """Draw the initial frames."""
        plot_grid_size = (3, self.nb_retrieved)
        self.fig = plt.figure(figsize=(16, 9))

        # Create the painter.
        self.paint_frame = plt.subplot2grid(
            plot_grid_size, (0, 0), rowspan=2, colspan=2, title="Draw here!")
        self.paint_frame.painter = art.Painter(self.paint_frame,
                                               self.update_data)

        # Create the sculptor.
        self.edit_frame = plt.subplot2grid(
            plot_grid_size, (0, 2), rowspan=2, colspan=2,
            title="Edit here!")
        self.edit_frame.sculptor = art.Sculptor(self.edit_frame,
                                                self.update_data)

        # Create the pickers.
        self.retrieved_frames = []
        for i in range(self.nb_retrieved):
            frame = plt.subplot2grid(plot_grid_size, (2, i), title=i+1)
            self.retrieved_frames.append(frame)
            frame.picker = art.ButtonDisplay(frame, self.update_data)
            frame.picker.text = frame.text(
                0.95, 0.01, frame.picker.params,
                verticalalignment='bottom',
                horizontalalignment='right',
                transform=frame.transAxes,
                fontsize=10)

        # Create the final display.
        self.show_frame = plt.subplot2grid(
            plot_grid_size, (0, 4), rowspan=2, colspan=2, title="Result")
        self.show_frame.display = SpiroDisplay(
            self.curve_type, self.show_frame, self.update_data)

        plt.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.95,
                            wspace=0.1, hspace=0.1)
    
    def update_matcher(self):
        """Update the matcher according to the curve type."""
        params = (0,) * self.samples.shape[1]
        self.matcher.get_curve = self.curve_type(*params).update_curve

#    def update_optimizer(self):
#        """Update the optimizer according to the curve type."""
#        params = (0,) * self.samples.shape[1]
#        self.optimizer.get_curve = self.curve_type(*params).reset_curve

    def update_data(self, from_ax):
        """Update data in frames in response to event."""
        if from_ax == self.paint_frame:
            self.transmit_painter_to_pickers()
        if from_ax in self.retrieved_frames:
            self.transmit_picker_to_sculptor(from_ax.picker)
            self.transmit_picker_to_display(from_ax.picker)
            self.transmit_picker_to_pickers(from_ax.picker)
        if from_ax == self.edit_frame:
            self.optimize_sculptor_curve()
            self.transmit_sculptor_to_pickers()
            self.transmit_sculptor_to_display()

    def transmit_painter_to_pickers(self):
        """Update pickers from hand drawing."""
        if self.paint_frame.lines:
            # Get hand-drawn curve.
            curve = np.array(self.paint_frame.lines[-1].get_data(orig=False))
            if curve.size <= 2:
                return
            # Retrieve closest matching curves.
            retrieved_args = (
                self.matcher(curve, self.samples)[:self.nb_retrieved, :])
            # Update pickers' plots.
            self.reset_pickers(retrieved_args)

    def transmit_picker_to_sculptor(self, picker):
        """Update sculptor data from selected picker."""
        scl = self.edit_frame.sculptor
        if picker.params is not None:
            # Perform deep copy of the params since they are actively modified in
            # the sculptor.
            params = picker.params.copy()
            data = picker.curve.get_data()
            sym_order = self.curve_type(*params).get_rot_sym_order()

            scl.reset(params, data, sym_order)
            scl.redraw()

    def transmit_picker_to_display(self, picker):
        """Update display data from selected picker."""
        if picker.params is not None:
            dsp = self.show_frame.display
            dsp.reset(picker.params, picker.curve.get_data())
            dsp.redraw()

# Put in a ButtonDisplayRow class?
    def transmit_picker_to_pickers(self, picker):
        """Update picker selection state."""
        for retf in self.retrieved_frames:
            if retf.picker.hold and retf.picker != picker:
                retf.picker.hold = False
                retf.set_axis_bgcolor(art.ButtonDisplay.bg_colors[0])
                retf.picker.redraw()

    def optimize_sculptor_curve(self):
        """Optimize the sculptor after curve editing."""
        scl = self.edit_frame.sculptor
        # Optimization
        if scl.to_opt and self.optimizer is not None:
            opt = self.optimizer
            params = scl.params
            cv_obj = self.curve_type(*params)
#            opt.get_curve = lambda p: cv_obj.update_curve(np.r_[params[:2], p])
#            
#            constr_dict = cv_obj.get_continuous_optimization_constraints()
#            opt.bounds = constr_dict.get('bounds')
#            opt.constr = constr_dict.get('constraints')
#            
#            print(constr_dict.get('bounds'))
#
#            res = opt.optimize(target_curve=scl.data, init_guess=params[2:],
#                               display=True)
#            
#            if res.success:
#                print('Successful optimization!')
#                print("Solution: ", res.x)
#
#            params[2:] = res.x
#            
#            fig = plt.figure(2)
#            ax = fig.add_subplot(111)
#            vals = np.linspace(opt.bounds[0][0], opt.bounds[0][1], 50)
#            energy = [opt._get_objective((val, params[3])) for val in vals]
#            ax.plot(vals, energy)
#            fig.show()            


            for idx in range(2,4):
                opt.get_curve = lambda p: cv_obj.update_curve(np.r_[params[:idx], p, params[idx + 1:]])
                cv_obj.update_curve(params)
                opt.bounds = cv_obj.get_bounds(idx)
    
                res = opt.optimize(target_curve=scl.data, init_guess=params[idx:idx + 1],
                                   display=True)
                
                params[idx] = res.x

#            fig = plt.figure(2)
#            ax = fig.add_subplot(111)
#            vals = np.linspace(opt.bounds[0], opt.bounds[1], 50)
#            energy = [opt._get_objective(val) for val in vals]
#            ax.plot(vals, energy)
#            fig.show()
            
#            if res.success:
#                print('Successful optimization!')
#                print("Solution: ", res.x)
#
#            params[idx] = res.x


            curve_pts = cv_obj.update_curve(params)
            scl.reset(params, curve_pts, scl.sym_order)
            scl.redraw()

    def transmit_sculptor_to_pickers(self):
        """Update pickers from curve editing."""
        scl = self.edit_frame.sculptor
        params = scl.params
        if params is None:
            return
        # Get closest curves.
        subspace = (self.samples[:, 0] == params[0]) * (self.samples[:, 1] == params[1])
        print(self.samples[subspace])
        dists2 = abs(self.samples[subspace][:, 2] - params[2])
        dists3 = abs(self.samples[subspace][:, 3] - params[3])
        ids2 = np.argsort(dists2)
        ids3 = np.argsort(dists3)
        closest_args2 = self.samples[subspace][ids2[:self.nb_retrieved/2]][::-1]
        closest_args3 = self.samples[subspace][ids3[:self.nb_retrieved/2]][::-1]
        self.reset_pickers(np.vstack([closest_args3, closest_args2]))
        
#        dists = ((self.samples - params) ** 2).sum(axis=1)
#        ids = np.argsort(dists)
#        closest_args = self.samples[ids[:self.nb_retrieved]]
        # Update pickers' plots.
#        self.reset_pickers(closest_args)

    def transmit_sculptor_to_display(self):
        """Update display data from curve editing."""
        scl = self.edit_frame.sculptor
        if scl.params is not None:
            dsp = self.show_frame.display
            dsp.reset(scl.params, scl.curve.get_data())
            dsp.redraw()

# put in a ButtonDisplayRow class?
    def reset_pickers(self, params):
        """Update display of the pickers."""
        # Update pickers' plots.
        for (par, frame) in zip(params, self.retrieved_frames):
            curve_pts = self.curve_type(*par).get_curve()
            frame.picker.reset(par, curve_pts)
            frame.picker.text.set_text(par)
            frame.picker.redraw()
#        self.subdivide_pickers(2)
    
    def subdivide_pickers(self, n):
        """Subdivide the pickers."""
        subdiv = []
        for i in range(n):
            phantom = self.fig.add_axes((i/n,0.03,1/n,0.33))
            phantom.xaxis.set_visible(False)
            phantom.yaxis.set_visible(False)
            phantom.set_zorder(-1)
            phantom.patch.set_alpha(0.5)
            phantom.patch.set_color('grey')

            subdiv.append(phantom)
        self.fig.canvas.draw()


class SpiroDisplay(art.AnimDisplay):
    """Specialization of AnimDisplay to animate Spirograph machines."""

    def __init__(self, curve_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curve_type = curve_type
        self.shapes = []
        self.text = None
        
        self.init_draw()

    def reset(self, params, data):
        """Reset the data."""
        super().reset(params, data)

# TODO: Find a better general way.
        params = self.params.copy()
        params[-1] = 0.
        curve_obj = self.curve_type(*params)
        # Positions (and angles).
        self.shapes[1].positions = curve_obj.get_curve()[:, :-1]
        if self.curve_type is cg.RouletteEllipseInCircle:
            self.shapes[1].angles = curve_obj.get_angles()[:-1] * 180 / np.pi
        self.shapes[2].positions = self.data
        # Design parameters.
        self.shapes[0].radius = params[0]
        if self.curve_type is cg.Hypotrochoid:
            self.shapes[1].radius = params[1]
        elif self.curve_type is cg.RouletteEllipseInCircle:
            semiaxes = curve_obj.get_ellipse_semiaxes()
            self.shapes[1].width = 2 * semiaxes[0]
            self.shapes[1].height = 2 * semiaxes[1]
        self.shapes[2].radius = params[1] / 20
        # Initial position (and angle).
        self.shapes[1].center = self.shapes[1].positions[:, 0]
        if self.curve_type is cg.RouletteEllipseInCircle:
            self.shapes[1].angle = 0.
        self.shapes[2].center = self.shapes[2].positions[:, 0]

        self.text.set_text(self.params)

        # Override the axes rescaling, taking the shapes into account.
        xmax = (self.shapes[0].radius,
                self.shapes[1].width / 2 + self.shapes[1].center[0],
                self.shapes[2].radius / 2 + self.shapes[2].center[0])
        dim = 1.1 * max(xmax)
        self.ax.set_xlim(-dim, dim)
        self.ax.set_ylim(-dim, dim)

    def init_draw(self):
        """Draw the spirograph corresponding to its arguments."""
        ax = self.ax
        self.ax.set_axis_bgcolor('lightgrey')
# TODO: Find a better general way.
        out_gear = mpat.Circle((0, 0), 0, edgecolor='r', facecolor='w', fill=True)
        if self.curve_type is cg.Hypotrochoid:
            int_gear = mpat.Circle((0, 0), 0, color='g', fill=True, alpha=0.7)
        elif self.curve_type is cg.RouletteEllipseInCircle:
            int_gear = mpat.Ellipse((0, 0), 0, 0, color='g', fill=True, alpha=0.7)
        tracer = mpat.Circle((0, 0), 0,  edgecolor='g', facecolor='w', fill=True)

        patches = (out_gear, int_gear, tracer)
        moving = (False, True, True)
# end of problematic part.

        for pat, mov in zip(patches, moving):
            shape = ax.add_artist(pat)
            shape.moving = mov
            self.shapes.append(shape)

        self.text = ax.text(0.95, 0.01, self.params,
                            verticalalignment='bottom',
                            horizontalalignment='right',
                            transform=ax.transAxes,
                            fontsize=11)

    def animate(self):
        """Create the current frame."""
        idx = self.time % self.nb_frames
        for shape in self.shapes:
            if shape.moving:
                nb_frames = shape.positions.shape[1]
                idx_ = idx if nb_frames == self.nb_frames else idx % nb_frames
                shape.center = shape.positions[:, idx_]
                # shape.angle = ...
# TODO: Find a better general way.
        if self.curve_type is cg.RouletteEllipseInCircle:
            idx_ = idx % self.shapes[1].angles.size
            self.shapes[1].angle = self.shapes[1].angles[idx_]
# end of problematic part.
        super().animate()


parts = (
    {'type': 'circle',
     'behavior': 'still'
    },
    {'type': 'circle',
     'behavior': 'moving'
    })


constraints = (
    {'type': 'inside',
     'left_operand': parts[1],
     'right_operand': parts[0]
    },
    {'type': 'rolling_without_slipping_on',
     'left_operand': parts[0],
     'right_operand': parts[1]
    })


def main():
    """Entry point."""
    plt.ioff()
    np.set_printoptions(precision=2, suppress=True)

    init_curve_type = cg.RouletteEllipseInCircle
#    init_curve_type = cg.Hypotrochoid
    distance = cdist.DistanceField()
    # For both the matcher and the optimizer, the curve function will be 
    # specified by Spirou.
    matcher = cmat.CurveMatcher(distance.get_dist, get_curve=None)
    optimizer = cmat.CurveOptimizer(distance.get_dist, get_curve=None)

    Spirou(init_curve_type, matcher, optimizer)
    plt.show()


if __name__ == "__main__":
    main()
