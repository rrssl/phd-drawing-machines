#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sampling the design space.

@author: Robin Roussel
"""
import numpy as np
import context
from mecha import SingleGearFixedFulcrumCDM as CDM

def fun():
    dom = list(CDM.ConstraintSolver.sample_feasible_domain())
    return np.array(dom)

def load(filename):
    dom = np.load(filename)
    return dom

if __name__ == "__main__":
    dom = fun()
    print("Number of samples: ", dom.shape[0])
    constraints = CDM.ConstraintSolver.get_constraints()
    if all((cons(dom.T) >= 0).all() for cons in constraints):
        print("All samples satisfy the constraints.")
    unique = np.vstack({tuple(row) for row in dom})
    if unique.shape == dom.shape:
        print("All samples are unique.")

    filename = "cdm_dom.npy"
    np.save(filename, dom)

    dom2 = load(filename)
    if np.allclose(dom, dom2):
        print("Samples were successfully saved.")
