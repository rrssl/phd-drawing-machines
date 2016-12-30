data = (
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
