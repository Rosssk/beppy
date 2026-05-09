import sympy as sm
import sympy.physics.mechanics as me

me.init_vprinting() 


M, m, l, k1, k2, c1, g, h, w, d, r = sm.symbols('M, m, l, k1, k2, c1, g, h, w, d, r')
q1, q2 = me.dynamicsymbols('q1 q2')
q1d = me.dynamicsymbols('q1', 1)

N = me.ReferenceFrame('N')
B = N.orientnew('B', 'axis', (q2, N.z))

O = me.Point('O')
block_point = O.locatenew('block', q1 * N.y)
pendulum_point = block_point.locatenew('pendulum', l * B.y)

O.set_vel(N, 0)
block_point.set_vel(N, q1d * N.y)
pendulum_point.v2pt_theory(block_point, N, B)

I_block = M / 12 * me.inertia(N, h**2 + d**2, w**2 + d**2, w**2 + h**2)
I_pendulum = 2*m*r**2/5*me.inertia(B, 1, 0, 1)


