spiro_data = (
    (0,
     {'valmin': 1,
      'valmax': 15,
      'valinit': 8,
      'label': "Outer gear radius"}),
    (1,
     {'valmin': 1,
      'valmax': 15,
      'valinit': 5,
      'label': "Inner gear radius"}),
    (2,
     {'valmin': 0.,
      'valmax': 15.,
      'valinit': 2.,
      'label': "Hole distance"})
    )

ellip_data = (
    (0,
     {'valmin': 1,
      'valmax': 15,
      'valinit': 8,
      'label': "Outer gear radius"}),
    (1,
     {'valmin': 1,
      'valmax': 15,
      'valinit': 5,
      'label': "Inner gear equiv. radius"}),
    (2,
     {'valmin': 0.,
      'valmax': 1.,
      'valinit': .2,
      'label': "Squared eccentricity"}),
    (3,
     {'valmin': 0.,
      'valmax': 15,
      'valinit': 2.,
      'label': "Hole distance"})
    )

cdm_data = (
    (0,                     # Radius of the turntable.
     {'valmin': 1,
      'valmax': 25,
      'valinit': 4,
      'label': "Turntable\nradius"}),
    (1,                     # Radius of the gear.
     {'valmin': 1,
      'valmax': 20,
      'valinit': 3,
      'label': "Gear radius"}),
    (2,                     # Distance from origin to fulcrum.
     {'valmin': 0.,
      'valmax': 20.,
      'valinit': 4.9,
      'label': "Fulcrum dist"}),
    (3,                     # Polar angle of the gear center.
     {'valmin': 0.,
      'valmax': 3.1415,
      'valinit': 2.8, #2 * math.pi / 3,
      'label': "Fulcrum-gear\nangle"}),
    (4,                     # Distance from fulcrum to penholder.
     {'valmin': 0.,
      'valmax': 20.,
      'valinit': 4.5,
      'label': "Fulcrum-\npenholder\ndist"}),
    (5,                     # Distance from gear center to slider.
     {'valmin': 0.,
      'valmax': 15.,
      'valinit': 1.8,
      'label': "Gear-slider dist"})
    )

hoot_data = (
    (0,                     # Radius of the turntable.
     {'valmin': 1,
      'valmax': 20,
      'valinit': 10,
      'label': "Turntable radius"}),
    (1,                     # Radius of gear 1.
     {'valmin': 1,
      'valmax': 20,
      'valinit': 4,
      'label': "Gear 1 radius"}),
    (2,                     # Radius of gear 2.
     {'valmin': 1,
      'valmax': 20,
      'valinit': 2,
      'label': "Gear 2 radius"}),
    (3,                     # Polar angle between gears.
     {'valmin': 0.,
      'valmax': 3.1415,
      'valinit': 3.1416 / 3.,
      'label': "Gear 2 angle"}),
    (4,                     # Distance from gear 1 center to pivot.
     {'valmin': 0.,
      'valmax': 20.,
      'valinit': 2.5,
      'label': "Pivot 1 radius"}),
    (5,                     # Distance from gear 2 center to pivot.
     {'valmin': 0.,
      'valmax': 20.,
      'valinit': 1.5,
      'label': "Pivot 2 radius"}),
    (6,                     # Length of arm 1.
     {'valmin': 0.,
      'valmax': 40.,
      'valinit': 10.,
      'label': "Arm 1 length"}),
    (7,                     # Length of arm 2.
     {'valmin': 0.,
      'valmax': 40.,
      'valinit': 8.,
      'label': "Arm 2 length"})
    )

kick_data = (
    (0,                     # Radius of gear 1.
     {'valmin': 1.,
      'valmax': 10.,
      'valinit': 4.,
      'label': "Gear 1 radius"}),
    (1,                     # Radius of gear 2.
     {'valmin': 1.,
      'valmax': 10.,
      'valinit': 2.,
      'label': "Gear 2 radius"}),
    (2,                     # Distance between gears.
     {'valmin': 0.,
      'valmax': 10.,
      'valinit': 8.,
      'label': "Distance between gears"}),
    (3,                     # Length of arm 1.
     {'valmin': 0.,
      'valmax': 10.,
      'valinit': 8.,
      'label': "Arm 1 length"}),
    (4,                     # Length of arm 2.
     {'valmin': 0.,
      'valmax': 10.,
      'valinit': 7.,
      'label': "Arm 2 length"}),
    (5,                     # Distance of end_effector from arm junction.
     {'valmin': 0.,
      'valmax': 10.,
      'valinit': 6.,
      'label': "End-effector distance"})
    )

import _context
from mecha import Thing
thing_data = [
    (i,                     # Amplitude
     {'valmin': 0.,
      'valmax': .5,
      'valinit': .15,
      'label': "a_{}".format(i)})
    for i in range(Thing.ConstraintSolver.nb_cprops)
    ]
