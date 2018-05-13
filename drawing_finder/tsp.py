# Copyright 2017, Gurobi Optimization, Inc.

# Solve a traveling salesman problem on a randomly generated set of
# points using lazy constraints.   The base MIP model only includes
# 'degree-2' constraints, requiring each node to have exactly
# two incident edges.  Solutions to this model may contain subtours -
# tours that don't visit every city.  The lazy constraint callback
# adds new constraints to cut them off.
import itertools
from gurobipy import GRB, Model, tuplelist, quicksum


def subtourelim(model, where, n):
    """Callback - use lazy constraints to eliminate sub-tours."""
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._vars)
        selected = tuplelist(
                (i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
        # find the shortest cycle in the selected edge list
        tour = subtour(n, selected)
        if len(tour) < n:
            # Add subtour elimination constraint for every pair of cities in
            # tour
            model.cbLazy(
                    quicksum(model._vars[i, j]
                             for i, j in itertools.combinations(tour, 2))
                    <= len(tour)-1)


def subtour(n, edges):
    """Given a tuplelist of edges, find the shortest subtour"""
    unvisited = list(range(n))
    cycle = range(n+1)  # initial length has 1 more city
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(cycle) > len(thiscycle):
            cycle = thiscycle
    return cycle


class TSPSolver:
    def __init__(self, num_points, distances):
        self.n = num_points

        m = Model()
        self.model = m
        m.Params.OutputFlag = 0

        # Create variables
        vars_ = m.addVars(
                distances.keys(), obj=distances, vtype=GRB.BINARY, name='e')
        for i, j in vars_.keys():
            vars_[j, i] = vars_[i, j]  # edge in opposite direction
        self.vars_ = vars_

        # You could use Python looping constructs and m.addVar() to create
        # these decision variables instead.  The following would be equivalent
        # to the preceding m.addVars() call...
        #
        # vars_ = tupledict()
        # for i,j in distances.keys():
        #   vars_[i,j] = m.addVar(obj=distances[i,j], vtype=GRB.BINARY,
        #                        name='e[%d,%d]'%(i,j))

        # Add degree-2 constraint
        m.addConstrs(vars_.sum(i, '*') == 2 for i in range(num_points))

        # Using Python looping constructs, the preceding would be...
        #
        # for i in range(n):
        #   m.addConstr(sum(vars_[i,j] for j in range(n)) == 2)

        m._vars = vars_
        m.Params.lazyConstraints = 1

    def solve(self):
        m = self.model
        m.optimize(lambda model, where: subtourelim(model, where, n=self.n))
        cost = m.objVal

        vals = m.getAttr('x', self.vars_)
        selected = tuplelist(
                (i, j) for i, j in vals.keys() if vals[i, j] > 0.5)

        tour = subtour(self.n, selected)
        assert len(tour) == self.n
        return tour, cost


def main():
    import math
    import sys
    import random

    if len(sys.argv) < 2:
        print('Usage: tsp.py npoints')
        return
    n = int(sys.argv[1])

    # Create n random points

    random.seed(1)
    points = [(random.randint(0, 100), random.randint(0, 100))
              for i in range(n)]

    # Dictionary of Euclidean distance between each pair of points

    dist = {(i, j):
            math.sqrt(sum((points[i][k] - points[j][k])**2 for k in range(2)))
            for i in range(n) for j in range(i)
            }

    solver = TSPSolver(n, dist)
    tour, cost = solver.solve()

    print("\nOptimal tour: {}".format(tour))
    print("Optimal cost: {}\n".format(cost))


if __name__ == "__main__":
    main()
