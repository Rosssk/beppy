from typing import Any

from numpy._typing._array_like import NDArray

from rbm import PRBM
import numpy as np
from tqdm import tqdm

mm = 1e-3
cm = 1e-2
t = 1e-3
A = np.pi*t**2
E = 1650e6 # Pa
I = np.pi*np.pi*t**4/2

def init_rbm():
    p = PRBM(3)

    p.add_body('A', np.array([0, 0, 0]))
    p.add_body('B', np.array([0, 0, 15*mm]))
    p.add_body('C', np.array([0, 0, 3*cm]))

    n = 3
    r = 14*mm

    for i in range(n):
        p.add_flexure('A', np.array((r*np.cos(i/n*2*np.pi), r*np.sin(i/n*2*np.pi), 0)),
                      'B', np.array((r*np.cos((i + 1)/n*2*np.pi), r*np.sin((i + 1)/n*2*np.pi), 0)))
        p.add_flexure('C', np.array((r*np.cos((i - .5)/n*2*np.pi), r*np.sin((i - .5)/n*2*np.pi), 0)),
                      'B', np.array((r*np.cos((i + .5)/n*2*np.pi), r*np.sin((i + .5)/n*2*np.pi), 0)))

    return p

X = []
Y = []

def add_points():
    # force_vecs= np.linspace([0, 0, 0], [100, 100, 100], 10)
    force_vecs = np.mgrid[0:100.1:5, 0:50.1:5, 0:50.1:5].reshape(3,-1).T

    print(force_vecs)
    for force_vec in tqdm(force_vecs):
        p = init_rbm()

        p.add_force('C', force_vec)
        p.solve_pose(["B", "C"], A, E, I)

        global X, Y
        X.append(force_vec)
        Y.append(p.solution.x)



add_points()
