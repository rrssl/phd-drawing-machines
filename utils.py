# -*- coding: utf-8 -*-
"""
Utility functions (only generators at this point).

@author: Robin Roussel
"""


def farey(n, asc=True):
    """Get a generator of the nth Farey sequence (ascending or descending)."""
    # See farey_bench.py for sources and comparison with other implementations.
    if asc:
        a, b, c, d = 0, 1,  1, n
    else:
        a, b, c, d = 1, 1, n-1, n
    yield (a, b)
    while (asc and c <= n) or (not asc and a > 0):
        k = int((n + b)/d)
        a, b, c, d = c, d, k*c - a, k*d - b
        yield (a, b)


def skipends(itr):
    """Adapt a generator to ignore the first and last item."""
    # Source: http://stackoverflow.com/a/2429118
    itr = iter(itr)  # Ensure we have an iterator
    next(itr)  # Skip the first
    prev = next(itr)
    for item in itr:
        yield prev
        prev = item
