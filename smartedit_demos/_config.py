fixkrv_data = {
    'disc_prop': (5, 3),
    'cont_prop': (.2, 1.),
    'init_poi_id': 0,
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
    'init_poi_id': (150, 117),
#    'init_poi_id': (53, 267),
#    'pts_per_dim': 20,
#    'keep_ratio': .05,
#    'deg_invar_poly': 2,
#    'nb_crv_pts': 2**6
    }

fixpos_data = {
    'disc_prop': (7, 3),
    'cont_prop': (.2, 1.),
    'init_poi_id': 0,
#    'pts_per_dim': 20,
#    'keep_ratio': .05,
#    'deg_invar_poly': 2,
#    'nb_crv_pts': 2**6
    }

fixisectangle_data = {
    'disc_prop': (5, 3),
    'cont_prop': (.31, .4755), # Quasi zero angle between segments
#    'init_poi_id': (11, 117),
    'init_poi_id': (53, 267),
    'pts_per_dim': 10,
#    'keep_ratio': .05,
#    'deg_invar_poly': 2,
#    'nb_crv_pts': 2**6
    }

fixisectpos_data = {
    'disc_prop': (5, 3),
    'cont_prop': (.31, .48), # Quasi zero angle between segments
#    'init_poi_id': (11, 117),
    'init_poi_id': (53, 267),
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
#    'props': (10, 4, 2, .69, 2.52, .86, 10.22, 9.56),
#    'init_poi_id': 100,

#    'props': (9, 5, 3, .96, .75, 1.5, 9.9, 7.9),
#    'init_poi_id': 100,

#    'props': (15, 14, 3, 1.3316335903703507, 3.7699182132015086, 2.0515806228896993, 32.6563894503349, 16.11314752952731),
#    'init_poi_id': 179,

#    'props': (15, 14, 3, 1.434072154053342, 0.7423780704389515, 2.820771399288182, 27.554290816613886, 27.051378347889496),
#    'init_poi_id': 150,

#    'props': (10, 6, 11, 1.4853902905536696, 3.088235294117645, 1.1274509803921546, 20.882352941176478, 20.392156862745097),
#    'init_poi_id': 150,

#    'props': (11, 2, 10, 1.8706271873653646, 1.470588235294116, 4.264705882352942, 13.72549019607844, 24.411764705882348),
#    'init_poi_id': 430,

#    'props': (9, 3, 8, 1.2399355754301493, 0.6254240033864922, 4.1852042394987965, 8.918108992638306, 16.215350573958347),
#    'init_poi_id': 60,

#    'props': (9, 5, 1, 0.727900216871495, 1.1158145805736175, 0.4587749862471635, 10.014880373310376, 7.612344891906315),
#    'init_poi_id': 163,

#    'props': (15, 10, 12, 1.7031039041476046, 9.03573732575494, 0.1854022988483206, 20.844958592298546, 32.25205504046563),
#    'init_poi_id': 44,

#    'props': (15, 2, 5, 1.3551568627450976, 1.6666666666666643, 3.4313725490196063, 18.72549019607844, 23.431372549019613),
#    'init_poi_id': 34,
#
#    'props': (14, 7, 12, 1.3056840861311134, 2.7781977344638142, 3.2218345961431583, 19.212580532222297, 18.07119693817469),
#    'init_poi_id': 471,

#    'props': (11, 4, 8, 2.6663151260504234, 3.5336518278970686, 1.3621295024499336, 17.481143343559438, 21.408257129610917),
#    'init_poi_id': 326,

    'props': (15, 15, 11, 1.4828358173458094, 6.662438560329804, 4.188996353258297, 34.33010940225151, 30.81963260425178),
    'init_poi_id': 581,


#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
    'ndim_invar_space': 3,
    'nb_crv_pts': 2**7
    }

fixlinehoot_data = {
    'props': (10, 4, 2, 1., 2.5, 1.5, 10., 8.),
    'init_poi_id': 0,
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
    'ndim_invar_space': 4,
    'nb_crv_pts': 2**7
    }
