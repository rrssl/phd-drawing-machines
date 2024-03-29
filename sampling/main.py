#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sampling the design space.

@author: Robin Roussel
"""
import time
import numpy as np
import _context

if(0):
    from mecha import EllipticSpirograph as TYPE
    filename = "ellip_dom.npy"
elif(0):
    from mecha import SingleGearFixedFulcrumCDM as TYPE
    filename = "cdm_dom.npy"
else:
    from mecha import HootNanny as TYPE
    filename = "hoot_dom.npy"
    # max_nb_turns = 3, grid_resol = 4; Nb. samples = 3988; Elapsed time: 19s

MAX_DVAL = 7
CONT_DENSITY = 4

#import warnings
#warnings.filterwarnings("error")

def fun():
    TYPE.ConstraintSolver.max_nb_turns = 7
    grid_size = (CONT_DENSITY,) * TYPE.ConstraintSolver.nb_cprops
    t1 = time.time()
    dom = list(TYPE.ConstraintSolver.sample_feasible_domain(grid_size))
    t2 = time.time()
    fields = TYPE.__name__, (MAX_DVAL, CONT_DENSITY), len(dom), t2-t1
    with open("sample_log", 'a') as log_file:
        print("{}; {}; {}; {}".format(*fields), file=log_file)
    print("Elapsed time: {}s".format(t2-t1))
    return np.array(dom)

def load(filename):
    dom = np.load(filename)
    return dom

if __name__ == "__main__":
    if(1):
        dom = fun()
        print("Number of samples: ", dom.shape[0])
        constraints = TYPE.ConstraintSolver.get_constraints()
        valid = np.array([(cons(dom.T) >= 0).all() for cons in constraints])
        if all(valid):
            print("All samples satisfy the constraints.")
        else:
            print("The following constraints are not satisfied: {}".format(
                  np.where(~valid)[0]))
        unique = np.vstack({tuple(row) for row in dom})
        if unique.shape == dom.shape:
            print("All samples are unique.")
        else:
            print("There are {} duplicate samples.".format(
                  dom.shape[0] - unique.shape[0]))

        np.save(filename, dom)
        dom2 = load(filename)
        if np.allclose(dom, dom2):
            print("Samples were successfully saved.")

    else:
        dom = load(filename)
        print(dom.shape)
        constraints = TYPE.ConstraintSolver.get_constraints()
        valid = np.array([(cons(dom.T) >= 0).all() for cons in constraints])
        invalid = np.where(~valid)[0]
        print(invalid)

        for id_ in invalid:
            print("Constraint no. {}".format(id_))
            values = constraints[id_](dom.T)
            print("Invalid samples: {}".format(np.where(values < 0)[0]))
            print("Max constraint violation: {}".format(min(values)))

        unique = np.vstack({tuple(row) for row in dom})
        if unique.shape == dom.shape:
            print("All samples are unique.")
        else:
            print("There are {} duplicate samples.".format(
                  dom.shape[0] - unique.shape[0]))
            samples = [tuple(row) for row in dom]
#            doublons = {}
            for i, sample in enumerate(samples):
                nb = samples.count(sample)
                if nb > 1:
                    print(i, sample)
#                    doublons[sample] = i
#            print(doublons)
