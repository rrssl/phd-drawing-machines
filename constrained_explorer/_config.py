from functools import partial
import numpy as np

import _context
from curveproc import compute_curvature
import mecha
from poitrackers import get_corresp_krvmax, get_corresp_isect

### Feature getters
# TODO: once PoI have been harmonized, used invariants.py instead

def get_position(curve, param, poi):
    return poi

def get_curvature(curve, param, poi):
    return compute_curvature(curve)[param]

def get_angle(curve, param, poi):
    return np.arctan2(poi[1], poi[0])

def get_curvature_and_angle(curve, param, poi):
    return np.r_[compute_curvature(curve)[param] * 3e-2, np.arctan2(poi[1], poi[0])]
#    return np.r_[compute_curvature(curve)[param], np.arctan2(poi[1], poi[0])]

def get_isect_angle(curve, param, poi):
    if poi is None or not poi.size:
        feats= np.full(2, 1e6)
    else:
        curve = curve[:, :-1] # Remove last point
        n = curve.shape[1]
        param = np.asarray(param)
        v = curve[:, (param+1)%n] - curve[:, param%n]
        v /= np.linalg.norm(v, axis=0)
        feats = v[:, 1] - v[:, 0]
    return feats

def get_dist(curve, param, poi):
    diff = poi[:, 1] - poi[:, 0]
    return diff[0]**2 + diff[1]**2

def get_dist_and_krv_diff(curve, param, poi):
    krv = compute_curvature(curve)[param]
    diffpos = (poi[:, 1] - poi[:, 0]) / poi[:, 0]
    diffkrv = (krv[1] - krv[0]) / krv[0]
    return diffpos[0]**2 + diffpos[1]**2 + diffkrv**2

### EllipticSpirograph

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
#    'cont_prop': (.31, .48), # Nonzero dist between PoIs
    'cont_prop': (.3, 1.), # Nonzero dist between PoIs
#    'cont_prop': (.3, .692), # Quasi zero dist
#    'cont_prop': (.1, .2),  # Non zero dist
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


### SingleGearFixedFulcrumCDM

fixposcdm_data = {
    'mecha_type': mecha.SingleGearFixedFulcrumCDM,
    'props': (4, 3, 5., 2.6, 5.9, 2.5),
    'init_poi_id': 0,
    'get_corresp': get_corresp_krvmax,
    'get_features': get_position,
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
#    'ndim_invar_space': 2,
#    'nb_crv_pts': 2**6
    }

fixkrvcdm_data = {
    'mecha_type': mecha.SingleGearFixedFulcrumCDM,
    'props': (7, 3, 13., 2.8, 14., 2.),
    'init_poi_id': 0,
    'get_corresp': get_corresp_krvmax,
    'get_features': get_curvature,
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
    'ndim_invar_space': 3,
#    'nb_crv_pts': 2**6
    }

fixlinecdm_data = {
    'mecha_type': mecha.SingleGearFixedFulcrumCDM,
    'props': (6, 5, 7.65, 2.66, 9.63, 3.79),
    'init_poi_id': 50,
    'get_corresp': get_corresp_krvmax,
    'get_features': get_angle,
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
    'ndim_invar_space': 3,
#    'nb_crv_pts': 2**6
    }

fixkrvlinecdm_data = {
    'mecha_type': mecha.SingleGearFixedFulcrumCDM,
    'props': (12, 4, 13.9, 3., 17.5, 3.3),
#    'props': (12, 3, 13., 2.3, 14., 2.5),
    'init_poi_id': 0,
    'get_corresp': get_corresp_krvmax,
    'get_features': get_curvature_and_angle,
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
#    'ndim_invar_space': 2,
#    'nb_crv_pts': 2**6
    }

fixisectanglecdm_data = {
    'mecha_type': mecha.SingleGearFixedFulcrumCDM,
    'props': (3, 2, 3.5, 2.9, 3.7, 1.7),
    'init_poi_id': (46, 56),
    'get_corresp': partial(get_corresp_isect, loc_size=6),
    'get_features': get_isect_angle,
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
#    'ndim_invar_space': 2,
#    'nb_crv_pts': 2**6
    }


### HootNanny

fixposhoot_data = {
    'mecha_type': mecha.HootNanny,
#    'props': (10, 4, 2, .69, 2.52, .86, 10.22, 9.56),
#    'init_poi_id': 100,

#    'props': (9, 5, 3, .96, .75, 1.5, 9.9, 7.9),
#    'init_poi_id': 100,

    'props': (15, 14, 3, 1.3316335903703507, 3.7699182132015086, 2.0515806228896993, 32.6563894503349, 16.11314752952731),
    'init_poi_id': 179,

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

#    'props': (15, 15, 11, 1.4828358173458094, 6.662438560329804, 4.188996353258297, 34.33010940225151, 30.81963260425178),
#    'init_poi_id': 581,

#    'props': (14, 13, 3, 2.8999591131718314, 1.001741010640731, 1.525113033369887, 29.0775357017975, 18.28479547355801),
#    'init_poi_id': 4514,

    'get_corresp': get_corresp_krvmax,
    'get_features': get_position,
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
    'ndim_invar_space': 3,
    'nb_crv_pts': 2**7
    }

fixlinehoot_data = {
    'mecha_type': mecha.HootNanny,
    'props': (10, 4, 2, 1., 2.5, 1.5, 10., 8.),
    'init_poi_id': 0,
    'get_corresp': get_corresp_krvmax,
    'get_features': get_angle,
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
    'ndim_invar_space': 4,
    'nb_crv_pts': 2**7
    }

fixkrvhoot_data = {
    'mecha_type': mecha.HootNanny,
#    'props': (10, 4, 2, 1., 2.5, 1.5, 10., 8.),
#    'init_poi_id': 0,

#    'props': (15, 12, 4, 1.1384513524979374, 6.3386603226742295, 2.741616697186904, 25.01649911156195, 21.870353600819016),
#    'init_poi_id': 182,

    'props': (14, 13, 3, 2.8999591131718314, 1.001741010640731, 1.525113033369887, 29.0775357017975, 18.28479547355801),
    'init_poi_id': 4514,

    'get_corresp': get_corresp_krvmax,
    'get_features': get_curvature,
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
    'ndim_invar_space': 4,
    'nb_crv_pts': 2**7
    }

fixisectanglehoot_data = {
    'mecha_type': mecha.HootNanny,
#    'props': (14, 12, 3, 1.3777060049918841, 7.727921357222151, 2.787223195468503, 22.22926906611704, 17.813880263168564),
#    'init_poi_id': (214, 333),

#    'props': (15, 2, 12, 2.346815465892966, 0.2314888219438842, 2.8500829210852996, 20.5412621739713, 31.666402410020652),
#    'init_poi_id': (156, 283),

#    'props': (12, 12, 3, 1.2494177746242003, 1.7916600602505284, 2.4766132868241755, 23.979705089583035, 18.11954970667516),
#    'init_poi_id': (84, 94),

    'props': (10, 4, 2, 1.0472, 3.290441176470587, 0.533088235294116, 10.882352941176478, 9.558823529411768),
    'init_poi_id': (67, 191),

    'get_corresp': partial(get_corresp_isect, loc_size=6),
    'get_features': get_isect_angle,
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
    'ndim_invar_space': 4,
    'nb_crv_pts': 2**7
    }

fixdisthoot_data = {
    'mecha_type': mecha.HootNanny,
#    'props': (12, 12, 3, 1.2494177746242003, 1.7916600602505284, 2.4766132868241755, 23.979705089583035, 18.11954970667516),
#    'init_poi_id': (59, 120),

#    'props': (14, 7, 12, 1.3056840861311134, 2.7781977344638142, 3.2218345961431583, 19.212580532222297, 18.07119693817469),
#    'init_poi_id': (471, 148),

    'props': (15, 10, 2, 1.3433724430459493, 1.9058189313461327, 1.98, 18.500079276993844, 17.13017282384655),
    'init_poi_id': (79, 90),

    'get_corresp': get_corresp_krvmax,
    'get_features': get_dist,
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
    'ndim_invar_space': 4,
    'nb_crv_pts': 2**7
    }

fixdistkrvhoot_data = {
    'mecha_type': mecha.HootNanny,
    'props': (15, 10, 2, 1.3433724430459493, 1.9058189313461327, 1.98, 18.500079276993844, 17.13017282384655),
    'init_poi_id': (79, 90),

    'get_corresp': get_corresp_krvmax,
    'get_features': get_dist_and_krv_diff,
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
    'ndim_invar_space': 2,
    'nb_crv_pts': 2**7
    }

### Kicker

fixposkick_data = {
    'mecha_type': mecha.Kicker,
    'props': (4., 2., 8., 8., 7., 6.),
    'init_poi_id': 46,

    'get_corresp': get_corresp_krvmax,
    'get_features': get_position,
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
    'ndim_invar_space': 4,
    'nb_crv_pts': 2**6
    }

### Thing


nb_cprops = mecha.Thing.ConstraintSolver.nb_cprops

fixposthing_data = {
    'mecha_type': mecha.Thing,
    'props': [mecha.Thing.ConstraintSolver.get_bounds(None, 0)[1]*.9] * nb_cprops,
    'init_poi_id': 70,

    'get_corresp': get_corresp_krvmax,
    'get_features': get_position,
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
    'ndim_invar_space': nb_cprops - 2,
    'nb_crv_pts': 2**8
    }

fixkrvthing_data = {
    'mecha_type': mecha.Thing,
    'props': (0.09191176470588247, 0.1663602941176472, 0.08226102941176472, 0.020220588235294157, 0.38419117647058854),
    'init_poi_id': (69, 1396),
#    'init_poi_id': 69,

    'get_corresp': get_corresp_krvmax,
    'get_features': get_curvature,
#    'pts_per_dim': 5,
#    'keep_ratio': .05,
#    'nbhood_size': .1,
    'ndim_invar_space': nb_cprops - 2,
    'nb_crv_pts': 2**8
    }
