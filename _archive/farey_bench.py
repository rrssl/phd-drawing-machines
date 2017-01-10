# -*- coding: utf-8 -*-
"""
Different implementations of a Farey sequence generator.

@author: Robin Roussel
"""

# source: https://en.wikipedia.org/wiki/Farey_sequence#Next_term
def farey_1(n, asc=True):
    """Python function to print the nth Farey sequence, either ascending or descending."""
    if asc: 
        a, b, c, d = 0, 1,  1 , n
    else:
        a, b, c, d = 1, 1, n-1, n
    yield (a,b)
    while (asc and c <= n) or (not asc and a > 0):
        k = int((n + b)/d)
        a, b, c, d = c, d, k*c - a, k*d - b
        yield (a,b)

# source: https://code.activestate.com/recipes/496821-farey-sequence/
def farey_2(limit):
    '''Fast computation of Farey sequence as a generator'''
    # n, d is the start fraction n/d (0,1) initially                            
    # N, D is the stop fraction N/D (1,1) initially                             
    pend = []
    n = 0
    d = N = D = 1
    while True:
        mediant_d = d + D
        if mediant_d <= limit:
            mediant_n = n + N
            pend.append((mediant_n, mediant_d, N, D))
            N = mediant_n
            D = mediant_d
        else:
            yield n, d
            if pend:
                n, d, N, D = pend.pop()
            else:
                break

# source: https://www.quora.com/What-are-the-fastest-algorithms-for-generating-coprime-pairs
def farey_3(n, a=1, b=1):
    """Generates all relatively prime pairs <= n. The larger number comes first."""
    yield (a,b)
    k = 1
    while a*k+b <= n:
        for i in farey_3(n, a*k+b, a):
            yield i
        k += 1

# source: https://code.activestate.com/recipes/496821-farey-sequence/
def farey_4(limit, start=(0, 1), stop=(1, 1)):
    '''recursive definition of a Farey sequence generator'''
    n, d = start
    N, D = stop
    mediant_d = d + D
    if mediant_d <= limit:
        mediant = (n + N), mediant_d
        for pair in farey_4(limit, start, mediant): yield pair
        for pair in farey_4(limit, mediant, stop): yield pair
    else:
        yield start


print([(a, b) for a, b in farey_1(5)])
print([(a, b) for a, b in farey_2(5)])
print([(a, b) for a, b in farey_3(5)])
print([(a, b) for a, b in farey_4(5)])