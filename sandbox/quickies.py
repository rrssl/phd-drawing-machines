import fractions
import numpy as np

def farey(n, asc=True):
    """Get a generator of the nth Farey sequence (ascending or descending)."""
    # See farey_bench.py for sources and comparison with other implementations.
    if asc:
        a, b, c, d = 0, 1, 1, n
    else:
        a, b, c, d = 1, 1, n-1, n
    yield (a,b)
    while (asc and c <= n) or (not asc and a > 0):
        k = int((n + b)/d)
        a, b, c, d = c, d, k*c - a, k*d - b
        yield (a,b)

def farlen_1(n):
    a, b, c, d = 0, 1,  1 , n
    tot = 1
    while c <= n:
        k = int((n + b)/d)
        a, b, c, d = c, d, k*c - a, k*d - b
        tot += 1
    return tot

def farlen_2(n):
    tot = 0
    if n > 0:
        for k in range(1, n + 1):
            if fractions.gcd(n, k) == 1:
                tot += 1
        tot += farlen_2(n - 1)
    else:
        tot += 1
    return tot

def skipends(itr):
    """Adapt a generator to ignore the first and last item."""
    # Source: http://stackoverflow.com/a/2429118
    itr = iter(itr)  # ensure we have an iterator
    next(itr)
    prev = next(itr)
    for item in itr:
        yield prev
        prev = item

def combi_1(n1, n2):
    cont_arr = [np.linspace(0, l * (n2 - 1) / n2, n2) for l in range(1, n1 + 1)]
    comb = []
    for a,b in skipends(farey(n1)):
        for c in cont_arr[b - 1]:
            comb.append([a, b, c])
    comb = np.array(comb)
    return comb

def combi_2(n1, n2):
    n_pairs = farlen_1(n1) - 1
    comb = np.empty((n_pairs * n2, 3))
    for i, (a,b) in enumerate(skipends(farey(n1))):
        for j, c in enumerate(np.linspace(0, b * (n2 - 1) / n2, n2)):
            comb[i * n2 + j, :] = (a, b, c)
    return comb

def combi_3(n1, n2):
    n_pairs = farlen_1(n1) - 1
    cont_arr = [np.linspace(0, l * (n2 - 1) / n2, n2) for l in range(1, n1 + 1)]
    comb = np.empty((n_pairs * n2, 3))
    for i, (a,b) in enumerate(skipends(farey(n1))):
        for j, c in enumerate(cont_arr[b - 1]):
            comb[i * n2 + j, :] = (a, b, c)
    return comb

def combi_g(n1, n2):
    cont_arr = [np.linspace(0, l * (n2 - 1) / n2, n2) for l in range(1, n1 + 1)]
    for a,b in skipends(farey(n1)):
        for c in cont_arr[b - 1]:
            yield a, b, c

#%%
from scipy.special import ellipe
R, S = 3, 2
ebound = lambda t: t * t + (S * np.pi / (2 * ellipe(t * t) * R)) ** 2 - 1

def ramanujan2(e):
    y = np.sqrt(1 -e)
    h = ((1 - y) / (1 + y)) ** 2
    return (np.pi / 4) * (1 + y) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))

ebound_approx = lambda t: t * t + (S * np.pi /
                                   (2 * ramanujan2(t * t) * R)) ** 2 - 1

def cantrell_ramanujan(e):
    y = np.sqrt(1 -e)
    h = ((1 - y) / (1 + y)) ** 2
    return (np.pi / 4) * (1 + y) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)) +
                                    (4 / np.pi - 14 / 11) * (h ** 12))

ebound_approx_2 = lambda t: t * t + (S * np.pi /
                                     (2 * cantrell_ramanujan(t * t) * R)) ** 2 - 1

def sykora(e):
    y = np.sqrt(1 - e)
    return ((np.pi * y + (1 - y) ** 2) / (1 + y)) - (1 / 8) * (y / (1 + y)) * (
        (1 - y) ** 2 / (np.pi * y + (1 + y) ** 2))

ebound_approx = lambda t: t * t + (S * np.pi / (2 * sykora(t * t) * R)) ** 2 - 1

#%%

def get_dist_1(cand, df):
    # Compute distance normalization factor.
    diag = (df.shape[0] * df.shape[0] + df.shape[1] * df.shape[1])**0.5
    normalization_factor = 0.05 * diag
    # Compute the average normalized distance.
    nb_samples = cand.shape[1]
    return sum(df[int(y), int(x)] / normalization_factor
               for x, y in cand.T) / nb_samples

def get_dist_2(cand, df):
    # Compute distance normalization factor.
    diag = (df.shape[0] * df.shape[0] + df.shape[1] * df.shape[1])**0.5
    normalization_factor = 0.05 * diag
    # Compute the average normalized distance.
    nb_samples = cand.shape[1]
    return df[tuple(cand[::-1].astype(np.intp))].sum() / (nb_samples * normalization_factor)

#%%
import curves as cu
import matplotlib.pyplot as plt
import matplotlib.patches as pat

rad = 1
inv = cu.CircleInvolute(rad, np.pi/4)
rng = np.linspace(0, 2 * np.pi)
curve = inv.get_point(rng)
circle = pat.Circle((0,0), rad, fill=False)

plt.plot(curve[0], curve[1])
plt.gca().add_patch(circle)
plt.gca().set_aspect('equal')
plt.show()

#%%
# https://en.wikipedia.org/wiki/Superformula
# Other possibility: http://math.stackexchange.com/a/788110
import matplotlib.pyplot as plt

phi = np.linspace(0, 2 * np.pi, 100)
rho = (abs(np.cos(3 * phi / 4))**6 + abs(np.sin(3 * phi / 4))**6)**(-1/6)

x = rho * np.cos(phi)
y = rho * np.sin(phi)
ax = plt.subplot(111)#, projection='polar')
ax.plot(x, y)
plt.gca().set_aspect('equal')
plt.show()

#%%
from mecha import EllipticSpirograph

# timeit: 1 loops, best of 3: 16.9 s per loop
def explore_samples_no_memo():
    m = EllipticSpirograph(8, 5, .2, .2)
    for combi in m.constraint_solver.sample_feasible_domain():
        m.reset(*combi)
        m.simulator.simulate_cycle(reuse=False)

# timeit: 1 loops, best of 3: 4.58 s per loop
def explore_samples_memo():
    m = EllipticSpirograph(8, 5, .2, .2)
    for combi in m.constraint_solver.sample_feasible_domain():
        m.reset(*combi)
        m.simulator.simulate_cycle(reuse=True)

#%%
import matplotlib.pyplot as plt
import numpy as np

e = .2
a = 1.
phi = np.linspace(0, 2 * np.pi, 100)
rho = a*(1 - e**2) / (1 - e*np.cos(3*phi))

x = rho * np.cos(phi)
y = rho * np.sin(phi)
ax = plt.subplot(111)#, projection='polar')
ax.plot(x, y)
plt.gca().set_aspect('equal')
plt.show()
#%%
import matplotlib
print(matplotlib.__version__)