import pychrono as chrono

# -------------------------------------------------
# System
# -------------------------------------------------
sys = chrono.ChSystemNSC()
sys.SetGravitationalAcceleration(chrono.ChVector3d(0, -9.81, 0))

# -------------------------------------------------
# Helper: simple box body
# -------------------------------------------------
def make_link_box(name, length, thickness, pos, fixed=False):
    body = chrono.ChBodyEasyBox(length, thickness, thickness,  # x, y, z
                                1000,                          # density
                                True, True)                    # visualize, collide
    body.SetPos(pos)
    body.SetName(name)
    body.SetBodyFixed(fixed)
    sys.Add(body)
    return body

# -------------------------------------------------
# Bodies: ground, link1 (R–P), link2 (P–R)
# -------------------------------------------------
L1 = 0.05  # link lengths ~ flexure arm lengths
L2 = 0.05

ground = make_link_box("ground", 0.02, 0.01,
                       chrono.ChVectorD(0, 0, 0),
                       fixed=True)

link1 = make_link_box("link1", L1, 0.005,
                      chrono.ChVectorD(L1/2, 0, 0))

link2 = make_link_box("link2", L2, 0.005,
                      chrono.ChVectorD(L1 + L2/2, 0, 0))

# -------------------------------------------------
# Joint 1: Revolute (ground – link1)
#   This is the first pseudo-hinge of the flexure
# -------------------------------------------------
rev1 = chrono.ChLinkLockRevolute()
rev1_cs = chrono.ChCoordsysD(
    chrono.ChVectorD(0, 0, 0),              # joint location
    chrono.Q_from_AngZ(0)                   # hinge axis = Z
)
rev1.Initialize(ground, link1, rev1_cs)
sys.AddLink(rev1)

# -------------------------------------------------
# Joint 2: Prismatic (link1 – link2)
#   Slider axis along X, approximating axial compliance segment
# -------------------------------------------------
pris = chrono.ChLinkLockPrismatic()
pris_cs = chrono.ChCoordsysD(
    chrono.ChVectorD(L1, 0, 0),             # nominal joint location
    chrono.Q_from_AngZ(0)                   # prismatic axis = X
)
pris.Initialize(link1, link2, pris_cs)
sys.AddLink(pris)

# Optionally: limit the prismatic stroke (pseudo-flexure axial range)
# (uncomment if you want hard limits)
# pris.GetLimit_X().SetActive(True)
# pris.GetLimit_X().SetMin(-0.002)
# pris.GetLimit_X().SetMax( 0.002)

# -------------------------------------------------
# Joint 3: Revolute (link2 – ground or an output body)
#   Second pseudo-hinge of the flexure
# -------------------------------------------------
rev2 = chrono.ChLinkLockRevolute()
rev2_cs = chrono.ChCoordsysD(
    chrono.ChVectorD(L1 + L2, 0, 0),
    chrono.Q_from_AngZ(0)
)
rev2.Initialize(link2, ground, rev2_cs)
sys.AddLink(rev2)

# -------------------------------------------------
# Optional: add rotational stiffness at the R joints
#   (turning the kinematic RPR into a PRBM flexure)
#   You can replace this with your own torque law.
# -------------------------------------------------
k_theta = 0.5  # N·m/rad, example

def make_rot_spring(name, bodyA, bodyB, csys, k):
    # Simple linear torsional spring around Z using a callback
    class RotSpring(chrono.ChLinkRotSpringCB):
        def __init__(self, k):
            super().__init__()
            self.k = k
        def Evaluate(self, time, angle, ang_vel, link):
            # torque = -k * angle
            return -self.k * angle

    spring = RotSpring(k)
    spring.SetName(name)
    spring.Initialize(bodyA, bodyB, csys)
    sys.AddLink(spring)
    return spring

# Attach springs co-located with the revolute joints
spring1 = make_rot_spring("k_rev1", ground, link1, rev1_cs, k_theta)
spring2 = make_rot_spring("k_rev2", link2, ground, rev2_cs, k_theta)

# -------------------------------------------------
# Simple time stepping
# -------------------------------------------------
t_end = 1.0
dt = 1e-3

# Give link1 a small initial rotation to excite the flexure
link1.SetWvel_par(chrono.ChVectorD(0, 0, 5.0))  # rad/s about Z

while sys.GetChTime() < t_end:
    sys.DoStepDynamics(dt)

    # Example: read out pseudo-end-effector pose at link2 tip
    tip_pos = link2.TransformPointLocalToParent(chrono.ChVectorD(L2/2, 0, 0))
    # You can log tip_pos.x, tip_pos.y, etc. for PRBM validation

# Done
