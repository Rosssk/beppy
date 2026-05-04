from jactest import PRBM as TESTRBM
from rbm import PRBM
from numpy import pi, cos, sin

def main():
    p1 = PRBM(3)
    p2 = TESTRBM(3)

    for p in (p1, p2):
        mm = 1e-3
        cm = 1e-2
        h = 3*cm
        p.add_body('A', (0, 0, 0))
        p.add_body('B', (0, 0, h/2))
        p.add_body('C', (0, 0, h))

        t = mm
        A = pi*t**2
        E = 1650e6 # Pa
        I = pi*t**4/2

        n = 3
        r = 14*mm

        for i in range(n):
            p.add_flexure('A', (r*cos(i/n*2*pi), r*sin(i/n*2*pi), 0),
                            'B', (r*cos((i + 1)/n*2*pi), r*sin((i + 1)/n*2*pi), 0))
            p.add_flexure('C', (r*cos((i - .5)/n*2*pi), r*sin((i - .5)/n*2*pi), 0),
                            'B', (r*cos((i + .5)/n*2*pi), r*sin((i + .5)/n*2*pi), 0))

        p.add_force('C', (0, 0, -4))

        p.solve_pose('BC', A, E, I, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
        print(p.solution)
        p.show()

if __name__ == '__main__':
    main()