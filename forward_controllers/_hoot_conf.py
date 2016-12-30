data = (
    (0,                     # Radius of the turntable.
     {'valmin': 1,
      'valmax': 25,
      'valinit': 10,
      'label': "Turntable radius"}),
    (1,                     # Radius of gear 1.
     {'valmin': 1,
      'valmax': 20,
      'valinit': 4,
      'label': "Gear 1 radius"}),
    (2,                     # Radius of gear 2.
     {'valmin': 1,
      'valmax': 10,
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
