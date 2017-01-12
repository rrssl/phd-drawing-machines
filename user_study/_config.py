import _context
from mecha import HootNanny
from poitrackers import get_corresp_krvmax

cand_name = 'Robin'

tasks = (
    {
        'taskid': 1,
        'mecha_type': HootNanny,
#        'init_props': (10, 6, 11, 2.0204428, 1.88598097, 2.10636952, 19.69823673, 18.22990056),
        'target_props': (10, 6, 11, 1.4853902905536696, 3.088235294117645, 1.1274509803921546, 20.882352941176478, 20.392156862745097),
        'get_corresp': get_corresp_krvmax,
        'get_features': lambda curve, param, poi: poi,
        'init_poi_id': 150,
    #    'pts_per_dim': 5,
    #    'keep_ratio': .05,
    #    'nbhood_size': .1,
        'ndim_invar_space': 3,
        'pt_density': 2**7
        },
    )
