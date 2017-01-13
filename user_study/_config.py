import numpy as np

import _context
from curveproc import compute_curvature
from mecha import HootNanny
from poitrackers import get_corresp_krvmax, get_corresp_isect

cand_name = 'Robin'

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

def get_poi_dist(curve, param, poi):
    diff = poi[:, 1] - poi[:, 0]
    return diff
#    return diff[0]**2 + diff[1]**2

tasks = (
    {
        'taskid': 1,
        'mecha_type': HootNanny,
        'target_props': (10, 6, 11, 1.4853902905536696, 3.088235294117645, 1.1274509803921546, 20.882352941176478, 20.392156862745097),
        'get_corresp': get_corresp_krvmax,
        'get_features': lambda curve, param, poi: poi,
        'init_poi_id': 150,
    #    'pts_per_dim': 5,
    #    'keep_ratio': .05,
    #    'nbhood_size': .1,
        'ndim_invar_space': 5,
        'pt_density': 2**7
        },
    {
        'taskid': 2,
        'mecha_type': HootNanny,
        'target_props': (15, 12, 4, 1.1384513524979374, 6.3386603226742295, 2.741616697186904, 25.01649911156195, 21.870353600819016),
        'get_corresp': get_corresp_krvmax,
        'get_features': lambda curve, param, poi: compute_curvature(curve)[param],
        'init_poi_id': 182,
    #    'pts_per_dim': 5,
    #    'keep_ratio': .05,
    #    'nbhood_size': .1,
        'ndim_invar_space': 5,
        'pt_density': 2**7
        },
    {
        'taskid': 3,
        'mecha_type': HootNanny,
        'target_props': (14, 7, 12, 1.3056840861311134, 2.7781977344638142, 3.2218345961431583, 19.212580532222297, 18.07119693817469),
        'get_corresp': get_corresp_krvmax,
        'get_features': get_poi_dist,
        'init_poi_id': (471, 148),
    #    'pts_per_dim': 5,
    #    'keep_ratio': .05,
    #    'nbhood_size': .1,
        'ndim_invar_space': 5,
        'pt_density': 2**7
        },
    {
        'taskid': 4,
        'mecha_type': HootNanny,
        'target_props': (14, 12, 3, 1.3777060049918841, 7.727921357222151, 2.787223195468503, 22.22926906611704, 17.813880263168564),
        'get_corresp': lambda ref_crv, ref_par, curves: get_corresp_isect(ref_crv, ref_par, curves, loc_size=10),
        'get_features': get_isect_angle,
        'init_poi_id': (214, 333),
    #    'pts_per_dim': 5,
    #    'keep_ratio': .05,
    #    'nbhood_size': .1,
        'ndim_invar_space': 5,
        'pt_density': 2**7
        },
    )
