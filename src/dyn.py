# Save as rp_r_prbm.py and run with python3
import pychrono as chrono
import math

# chrono.SetChronoDataPath("")   # optional

# System
sys = chrono.ChSystemNSC()

# Ground
ground = chrono.ChBody()
ground.SetBodyFixed(True)
sys.Add(ground)

# Parameters (meters, kg, N/m, N·m/rad)
L1 = 0.05
L2 = 0.05
m_link = 0.01
Izz = 1e-6
k_rot1 = 0.5    # torsional spring at R1 [N·m/rad]
k_rot2 = 0.5    # torsional spring at R2 [N·m/rad]
k_axial = 1000  # axial spring on P [N/m]
damping_rot = 0.01
damping_axial = 0.1

# Bodies
link1 = chrono.ChBody()
link1.SetPos(chrono.ChVectorD(L1/2, 0, 0))
link1.SetMass(m_link)
link1.SetInertiaXX(chrono.ChVectorD(1e-6,1e-6,Izz))
sys.Add(link1)

link2 = chrono.ChBody()
link2.SetPos(chrono.ChVectorD(L1 + L2/2, 0, 0))
link2.SetMass(m_link)
link2.SetInertiaXX(chrono.ChVectorD(1e-6,1e-6,Izz))
sys.Add(link2)

# Revolute R1: ground - link1 (z axis out of plane)
rev1 = chrono.ChLinkLockRevolute()
rev1.Initialize(ground, link1, chrono.ChCoordsysD(chrono.ChVectorD(0,0,0), chrono.QUNIT))
sys.Add(rev1)

# Prismatic P: link1 - link2, axis along x, rotations locked so moments are transmitted
prism = chrono.ChLinkLockPrismatic()
prism.Initialize(link1, link2, chrono.ChCoordsysD(chrono.ChVectorD(L1,0,0), chrono.QUNIT))
sys.Add(prism)

# Revolute R2: link2 - ground
rev2 = chrono.ChLinkLockRevolute()
rev2.Initialize(link2, ground, chrono.ChCoordsysD(chrono.ChVectorD(L1+L2,0,0), chrono.QUNIT))
sys.Add(rev2)

# Torsional springs at revolutes (ChLinkRotSpring)
rot_spring1 = chrono.ChLinkRotSpring()
rot_spring1.Initialize(ground, link1, chrono.ChCoordsysD(chrono.ChVectorD(0,0,0), chrono.QUNIT))
rot_spring1.SetSpringK(k_rot1)
rot_spring1.SetDamperR(damping_rot)
sys.Add(rot_spring1)

rot_spring2 = chrono.ChLinkRotSpring()
rot_spring2.Initialize(link2, ground, chrono.ChCoordsysD(chrono.ChVectorD(L1+L2,0,0), chrono.QUNIT))
rot_spring2.SetSpringK(k_rot2)
rot_spring2.SetDamperR(damping_rot)
sys.Add(rot_spring2)

# Axial spring on the prismatic DOF: use ChLinkSpring between two points on link1 and link2
axial = chrono.ChLinkSpring()
pA = chrono.ChVectorD(L1, 0, 0)
pB = chrono.ChVectorD(L1, 0, 0)  # local points coincide at joint
axial.Initialize(link1, link2, False, pA, pB)
axial.Set_SpringK(k_axial)
axial.Set_SpringR(damping_axial)
sys.Add(axial)

# Add a small tip mass and apply a static force for validation
tip = chrono.ChBody()
tip.SetMass(0.001)
tip.SetPos(chrono.ChVectorD(L1+L2, -0.02, 0))
tip.SetBodyFixed(False)
sys.Add(tip)
# attach tip rigidly to link2
fix = chrono.ChLinkLockLock()
fix.Initialize(link2, tip, chrono.ChCoordsysD(chrono.ChVectorD(L1+L2,0,0), chrono.QUNIT))
sys.Add(fix)

# Apply a constant downward force at tip via body force
tip_force = chrono.ChForce()
tip_force.SetMode(chrono.ChForce.FORCE)
tip_force.SetDir(chrono.ChVectorD(0,-1,0))
tip_force.SetMforce(0.1)  # 0.1 N downward
tip.AddForce(tip_force)

# Solver and integrator
sys.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
sys.SetMaxItersSolverSpeed(50)
sys.SetMaxItersSolverStab(50)

# Simulate a short static-like run
for i in range(4000):
    sys.DoStepDynamics(1e-4)

print("Final link2 angle (rad):", link2.GetRot().Q_to_AngAxis()[0])
