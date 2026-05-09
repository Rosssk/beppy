from sympy import symbols, sin, pi, cos, acos, Rational, simplify, atan2
from sympy.physics.mechanics import RigidBody, inertia, kinetic_energy
from sympy.physics.vector import init_vprinting, dynamicsymbols, ReferenceFrame, Point, Dyadic
from sympy.vector import Vector

init_vprinting()

# Global reference frame and origin
N = ReferenceFrame('N')
O = Point('O')
O.set_vel(N, 0*N.x)

def create_body(id, fixed, m, Ixx, Iyy, Izz, ic, x0 = 0, y0 = 0, z0 = 0):
    if fixed:
        P = O
        N_b = N
        P.set_vel(N, 0*N.x)
        q = []
    else:
        P = Point(f'P_b{id}')
        x, y, z, phi, theta, psi = dynamicsymbols(f'x{id} y{id} z{id} phi{id} theta{id} psi{id}')
        ic.update({
            x: x0, y: y0, z: z0, phi: 0, theta: 0, psi: 0
        })
        N_b = ReferenceFrame(f'N_b{id}')
        N_b.orient_body_fixed(N, (phi, theta, psi), 'XYZ')
        P.set_pos(O, x * N.x + y * N.y + z * N.z)
        P.set_vel(N, x.diff() * N.x + y.diff() * N.y + z.diff() * N.z)
        q = [x, y, z, phi, theta, psi]

    I = inertia(N_b, Ixx, Iyy, Izz)
    return RigidBody(f'B{id}', P, N_b, m, (I, P)), q

def get_body_attach(body, i, r):
    attach_local = Point(f'attach_{body.name.lower()}_{i}')
    attach_local.set_pos(body.masscenter, r*cos(i*Rational(2,3)*pi) * body.frame.x + r*sin(i*Rational(2,3)*pi) * body.frame.y)
    attach_local.v2pt_theory(body.masscenter, N, body.frame)
    return attach_local

def get_beam_energy(id, pos_a, body_a, pos_b, body_b, gamma, k_th, k_s, ic):
    # Fixed local offsets for the spring attachments
    beam_ab_local_a = pos_b.pos_from(pos_a).express(body_a.frame).subs(ic)
    beam_ba_local_b = pos_a.pos_from(pos_b).express(body_b.frame).subs(ic)

    spring_a = Point(f'spring_{id}_1_a')
    spring_a.set_pos(pos_a, beam_ab_local_a * gamma)
    spring_b = Point(f'spring_{id}_1_b')
    spring_b.set_pos(pos_b, beam_ba_local_b * gamma)

    # Set to stationary in respective body reference frame
    spring_a.v2pt_theory(pos_a, N, body_a.frame)
    spring_b.v2pt_theory(pos_b, N, body_b.frame)

    # Axial spring
    spring_vec = spring_b.pos_from(spring_a)
    length = spring_vec.magnitude()
    length0 = length.subs(ic)
    V_k = 0.5 * k_s * (length - length0)**2

    # Live spring axis and rigid link directions
    e_axial = spring_vec.normalize()
    link_a_hat = spring_a.pos_from(pos_a).normalize()  # const in body_a.frame
    link_b_hat = spring_b.pos_from(pos_b).normalize()  # const in body_b.frame

    # Geodesic angles via atan2(|cross|, dot) — smooth at rest
    dot_a = link_a_hat.dot(e_axial)
    cross_a = link_a_hat.cross(e_axial).magnitude()
    theta_a = atan2(cross_a, dot_a)

    dot_b = (-link_b_hat).dot(e_axial)
    cross_b = (-link_b_hat).cross(e_axial).magnitude()
    theta_b = atan2(cross_b, dot_b)

    V_th = 0.5 * k_th * (theta_a**2 + theta_b**2)

    return 0, V_k + V_th  # Beam mass not yet implemented


def main():
    V = 0 # Potential energy
    T = 0 # Kinetic energy

    g = symbols('g')
    gamma, k_th, k_s = symbols('gamma k_th k_s') # PRBM parameters
    E_mod, V_rat, rho = symbols('E_mod V_rat rho') # Material parameters
    h, r_f, r_b, t_f, t_b = symbols('h r_f r_b t_f t_b') # Geometric parameters

    ic = {}

    # Bodies are discs for now instead of equilateral triangles
    m_body = rho * t_b * pi * r_b**2
    Ixx_body = Iyy_body = m_body * r_b**2/4
    Izz_body = Ixx_body * 2

    B1, _ = create_body(1, True, m_body, Ixx_body, Iyy_body, Izz_body, ic)
    B2, q2 = create_body(2, False, m_body, Ixx_body, Iyy_body, Izz_body, ic, 0, 0, h/2)
    B3, q3 = create_body(3, False, m_body, Ixx_body, Iyy_body, Izz_body, ic, 0, 0, h)
    q = [*q2, *q3]

    # Add attachment points
    for i in range(3):
        attach_a = get_body_attach(B1, i, r_f)
        attach_b = get_body_attach(B2, (i + 1) % 3, r_f)
        attach_c = get_body_attach(B3, (i + 2) % 3, r_f)

        V_b1, T_b1 = get_beam_energy(f'{i}_1', attach_a, B1, attach_b, B2, gamma, k_th, k_s, ic)
        V_b2, T_b2 = get_beam_energy(f'{i}_2', attach_b, B2, attach_c, B3, gamma, k_th, k_s, ic)

        V += V_b1 + V_b2
        T += T_b1 + T_b2

    T += sum(kinetic_energy(N, body) for body in (B2, B3))

    V += -m_body * g * (B2.masscenter.pos_from(O).dot(N.z) +
                        B3.masscenter.pos_from(O).dot(N.z))

if __name__ == '__main__':
    main()