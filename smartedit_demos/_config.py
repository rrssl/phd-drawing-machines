fixkrv_data = {
    'disc_prop': (5, 3),
    'cont_prop': (.2, 1.),
    'pts_per_dim': 15,
#    'keep_ratio': .05,
#    'deg_invar_poly': 2,
#    'nb_crv_pts': 2**6
    }

fixdist_data = {
    'disc_prop': (5, 3),
#    'cont_prop' = (.31, .48) # Nonzero dist between PoIs
#    'cont_prop' = (.3, 1.) # Nonzero dist between PoIs
#    'cont_prop' = (.3, .692) # Quasi zero dist
    'cont_prop': (.1, .2),  # Non zero dist
#    'pts_per_dim': 20,
#    'keep_ratio': .05,
#    'deg_invar_poly': 2,
#    'nb_crv_pts': 2**6
    }

fixpos_data = {
    'disc_prop': (7, 3),
    'cont_prop': (.2, 1.),
#    'pts_per_dim': 20,
#    'keep_ratio': .05,
#    'deg_invar_poly': 2,
#    'nb_crv_pts': 2**6
    }

fixisectangle_data = {
    'disc_prop': (5, 3),
    'cont_prop': (.31, .4755), # Quasi zero angle between segments
    'pts_per_dim': 10,
#    'keep_ratio': .05,
#    'deg_invar_poly': 2,
#    'nb_crv_pts': 2**6
    }

fixisectpos_data = {
    'disc_prop': (5, 3),
    'cont_prop': (.31, .48), # Quasi zero angle between segments
    'pts_per_dim': 17,
#    'keep_ratio': .05,
    'deg_invar_poly': 3,
#    'nb_crv_pts': 2**6
    }

fixposcdm_data = {
    'disc_prop': (4, 3),
    'cont_prop': (5., 2.6, 5.9, 2.5),
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
#    'ndim_invar_space': 2,
#    'nb_crv_pts': 2**6
    }

fixkrvcdm_data = {
    'disc_prop': (7, 3),
    'cont_prop': (13., 2.8, 14., 2.),
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
    'ndim_invar_space': 3,
#    'nb_crv_pts': 2**6
    }

fixlinecdm_data = {
    'disc_prop': (6, 5),
    'cont_prop': (7.65, 2.66, 9.63, 3.79),
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
    'ndim_invar_space': 3,
#    'nb_crv_pts': 2**6
    }

fixkrvlinecdm_data = {
    'disc_prop': (12, 4),
#    'disc_prop': (12, 3),
    'cont_prop': (13.9, 3., 17.5, 3.3),
#    'cont_prop': (13., 2.3, 14., 2.5),
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
#    'ndim_invar_space': 2,
#    'nb_crv_pts': 2**6
    }

fixisectanglecdm_data = {
    'disc_prop': (2, 2),
    'cont_prop': (3.5, 2.9, 3.8, 1.7),
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
#    'ndim_invar_space': 2,
#    'nb_crv_pts': 2**6
    }

fixposhoot_data = {
#    'disc_prop': (10, 4, 2),
#    'disc_prop': (9, 5, 3),
    'disc_prop': (15, 14, 3),
#    'disc_prop': (10, 6, 11),
#    'cont_prop': (.69, 2.52, .86, 10.22, 9.56),
#    'cont_prop': (.96, .75, 1.5, 9.9, 7.9)
    'cont_prop': (1.3316335903703507, 3.7699182132015086, 2.0515806228896993, 32.6563894503349, 16.11314752952731),
#    'cont_prop': (1.434072154053342, 0.7423780704389515, 2.820771399288182, 27.554290816613886, 27.051378347889496),
#    'cont_prop': (1.4853902905536696, 3.088235294117645, 1.1274509803921546, 20.882352941176478, 20.392156862745097),
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
    'ndim_invar_space': 3,
    'nb_crv_pts': 2**7
    }

fixlinehoot_data = {
    'disc_prop': (10, 4, 2),
    'cont_prop': (1., 2.5, 1.5, 10., 8.),
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
    'ndim_invar_space': 3,
    'nb_crv_pts': 2**7
    }
