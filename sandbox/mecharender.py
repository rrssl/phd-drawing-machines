# -*- coding: utf-8 -*-
"""
Mechanism rendering module.

@author: Robin Roussel
"""
import math
import matplotlib.patches as pat
import matplotlib.transforms as mtrans

class RenderablePart:
    """Base interface for all renderable components."""
    
    def __init__(self, position=(0.,0.), orientation=0., color='grey', 
                 alpha=1.):
        self.color = color
        self.alpha = alpha
        
        self.position = position
        self.orientation = orientation # radians
        self.model = None
        self.init_model()
    
    def init_model(self):
        """Instantiate the graphical model to be rendered."""
        pass
    
    def update_model(self):
        """Update the graphical model with the current parameters."""
        pass


class Joint(RenderablePart):
    """Base joint."""
    
    def __init__(self, center, radius, *args, **kwargs):
        self.radius = radius
        if kwargs.get('alpha') is None:
            kwargs['alpha'] = 0.7
        super().__init__(center, *args, **kwargs)
        
    def init_model(self):
        """Instantiate the graphical model to be rendered."""
        self.model = pat.Circle(self.position, self.radius, color=self.color, 
                                alpha=self.alpha)
    
    def update_model(self):
        """Update the graphical model with the current parameters."""
        self.model.center = self.position
        self.model.radius = self.radius


class Hinge(Joint):
    """Hinged joint."""
    
    def __init__(self, *args, **kwargs):
        if kwargs.get('color') is None:
            kwargs['color'] = 'red'
        super().__init__(*args, **kwargs)


class Slider(Joint):
    """Prismatic joint."""

    def __init__(self, *args, **kwargs):
        if kwargs.get('color') is None:
            kwargs['color'] = 'green'
        super().__init__(*args, **kwargs)


class Penholder(Joint):
    """Penholder."""

    def __init__(self, *args, **kwargs):
        if kwargs.get('color') is None:
            kwargs['color'] = 'lightblue'
        super().__init__(*args, **kwargs)


class Gear(RenderablePart):
    """Spur gear."""

    def __init__(self, center, shape, params, external=True, *args, **kwargs):
        self.shape = shape
        self.params = params
        self.external = external
        super().__init__(center, *args, **kwargs)
        
    def init_model(self):
        """Instantiate the graphical model to be rendered."""
        if self.shape == 'circle':
            if self.external:
                radius = self.params
                self.model = pat.Circle(self.position, radius, 
                                        color=self.color, alpha=self.alpha)
        elif self.shape == 'ellipse':
            if self.external:
                width = 2 * self.params[0]
                height = 2 * self.params[1]
                angle = self.orientation * 180 / math.pi
                self.model = pat.Ellipse(self.position, width, height, angle,
                                         color=self.color, alpha=self.alpha)
        else:
            print('Unsupported shape type.')
            self.model = None
    
    def update_model(self):
        """Update the graphical model with the current parameters."""
        self.model.center = self.position
        if self.shape == 'circle':
            self.model.radius = self.radius
        elif self.shape == 'ellipse':
            self.model.angle = self.orientation * 180 / math.pi
            self.model.width = 2 * self.params[0]
            self.model.height = 2 * self.params[1]
            
class Rod(RenderablePart):
    """Rod."""
    
    def __init__(self, center, length, thickness, *args, **kwargs):
        self.length = length
        self.thickness = thickness
        super().__init__(center, *args, **kwargs)
        
    def init_model(self):
        """Instantiate the graphical model to be rendered."""
        width = self.length
        height = self.thickness
        angle = self.orientation * 180 / math.pi
        self.model = pat.Rectangle(self.position, width, height, angle, 
                                   color=self.color, alpha=self.alpha)
        
        self.init_angle = self.orientation # keep track for later updates
    
    def update_model(self):
        """Update the graphical model with the current parameters."""
        self.model.xy = self.position
        self.model.set_width(self.length)
        self.model.set_height(self.thickness)

        angle = self.orientation - self.init_angle
        rot = mtrans.Affine2D().rotate_around(
            self.position[0], self.position[1], angle)
        self.model.set_transform(rot)
