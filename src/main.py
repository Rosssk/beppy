import numpy as np

import mujoco
import mujoco.viewer
from scipy.interpolate import interp1d

def _body(parent, name, pos, r, t, rho, quat=None):
    vol = (np.sqrt(3) / 4 * (2 * r) ** 2) * t
    b = parent.add_body()
    b.name = name
    b.pos  = [float(v) for v in pos]
    b.mass = rho * vol
    I = max(0.4 * b.mass * r ** 2, 1e-12)
    b.inertia = [I, I, I]
    if quat is not None:
        b.quat = quat

    _site(b, f"center_{name}", [0,0,0], [1, 0, 0, 1])
    return b

def _ball(body, name, pos, k, d):
    """Add a 3-DOF ball joint with isotropic stiffness and damping."""
    j           = body.add_joint()
    j.name      = name
    j.type      = mujoco.mjtJoint.mjJNT_BALL
    j.pos       = pos
    j.stiffness = np.array([k, k, k])
    j.damping   = np.array([d, d, d])
    return j


def _site(body, name, pos, color=None):
    site = body.add_site()
    site.name = name
    site.pos = pos
    site.size = [1e-4, 1e-4, 1e-4]
    if color is None:
        color = [1.0, 0.8, 0.0, 1.0]
    site.rgba = color

def _free(body):
    j = body.add_joint()
    j.type = mujoco.mjtJoint.mjJNT_FREE
    return j


def main():
    # Parameters:
    d = 1e-4  # flexure diameter

    t = 5e-4  # platform thickness
    h = 2.5e-3  # module height

    r = 1e-3  # attachment radius
    E = 250e6  # Young's modulus
    v = 0.3  # Poisson's ratio
    rho = 1200  # density
    damping = 10

    # Compute values:
    h_inter_platform = (h - 3 * t) / 2
    L_flex = np.sqrt(h_inter_platform**2 + (2*r*np.sin(np.pi/6))**2)
    I0 = np.pi * (d ** 4) / 64
    A0 = np.pi * (d ** 2) / 4
    f_bg = I0 / (A0 * L_flex ** 2)  # Beam geometry factor

    # Table 2 data from https://www.researchgate.net/publication/305312934_Extension_Effects_in_Compliant_Joints_and_Pseudo-Rigid-Body_Models
    _gamma = np.array([0.205, 0.201, 0.187, 0.169])
    _kh = np.array([2.035, 2.032, 2.031, 2.024])
    _kex = np.array([0.224, 0.638, 0.978, 0.997])

    _fbg   = np.array([1e-4, 1e-3, 5e-3, 1e-2])
    _log_fbg = np.log10(_fbg)
    _gamma_interp = interp1d(_log_fbg, _gamma, kind='linear', fill_value='extrapolate')
    _kh_interp = interp1d(_log_fbg, _kh, kind='linear', fill_value='extrapolate')
    _kex_interp = interp1d(_log_fbg, _kex, kind='linear', fill_value='extrapolate')

    log_fbg = np.log10(f_bg)
    gamma = float(_gamma_interp(log_fbg))
    k_h_fac = float(_kh_interp(log_fbg))
    k_ex_fac = float(_kex_interp(log_fbg))

    # Now scale to physical units
    k_h_phys = k_h_fac * E * I0 / L_flex  # N·m/rad
    k_ex_phys = k_ex_fac * E * A0 / L_flex  # N/m

    print(f"k_h_fac  = {k_h_fac:.4f}  (dimensionless)")
    print(f"k_ex_fac = {k_ex_fac:.4f}  (dimensionless)")
    print(f"k_h_phys = {k_h_phys:.3e} N·m/rad")
    print(f"k_ex_phys= {k_ex_phys:.3e} N/m")

    k_h = k_h_phys
    k_ex = k_ex_phys

    m_flex = rho * A0 * L_flex

    spec = mujoco.MjSpec()
    spec.modelname = "prbm"
    spec.option.gravity = [0., 0., -9.81]
    spec.option.timestep = 2e-4

    body_a = _body(spec.worldbody, "body_a", [0, 0, 0], r, t,  rho)
    body_b = _body(spec.worldbody, "body_b", [0, 0, h / 2], r, t, rho)
    body_c = _body(spec.worldbody, "body_c", [0, 0, h], r, t, rho)

    _free(body_b)
    _free(body_c)

    n = 3
    for i in range(n):
        con_a = np.array(body_a.pos) + np.array([r*np.cos(i/n*2*np.pi), r*np.sin(i/n*2*np.pi), 0])
        con_b = np.array(body_b.pos) + np.array([r*np.cos((i + 1)/n*2*np.pi), r*np.sin((i + 1)/n*2*np.pi), 0])
        con_c = np.array(body_c.pos) + np.array([r*np.cos((i + 2)/n*2*np.pi), r*np.sin((i + 2)/n*2*np.pi), 0])

        v_ab = (con_b - con_a) * gamma + con_a - body_a.pos
        v_ba = (con_a - con_b) * gamma + con_b - body_b.pos
        v_bc = (con_c - con_b) * gamma + con_b - body_b.pos
        v_cb = (con_b - con_c) * gamma + con_c - body_c.pos

        _site(body_a, f"attach_a_{i}", con_a - body_a.pos)
        _site(body_b, f"attach_b_{i}", con_b - body_b.pos)
        _site(body_c, f"attach_c_{i}", con_c - body_c.pos)

        def _piv(body, name, pos):
            I_rod = m_flex/2 * (L_flex / np.sqrt(3)) ** 2
            p = body.add_body()
            p.name = name
            p.pos = pos
            p.mass = m_flex/2
            p.inertia = [I_rod, I_rod, I_rod]
            _ball(p, f"{name}_ball", [0, 0, 0], k_h_phys, damping)
            _site(p, f"{name}_c", [0, 0, 0], [0, 1, 0, 1])

        _piv(body_a, f"piv_ab_{i}", v_ab)
        _piv(body_b, f"piv_ba_{i}", v_ba)
        _piv(body_b, f"piv_bc_{i}", v_bc)
        _piv(body_c, f"piv_cb_{i}", v_cb)

        spring = spec.add_tendon()
        spring.name = f"radial_spring_ab_{i}"
        spring.stiffness = [k_ex_phys, 0, 0]
        spring.damping = [damping, 0, 0]
        spring.rgba        = [0.9, 0.1, 0.1, 0.6]
        spring.width       = d
        spring.wrap_site(f"piv_ab_{i}_c")
        spring.wrap_site(f"piv_ba_{i}_c")

        # Compute rest length as distance between the two pivot sites at assembly
        rest_len = float(np.linalg.norm(
            (np.array(body_b.pos) + v_ba) - (np.array(body_a.pos) + v_ab)
        ))
        spring.springlength = [rest_len, rest_len]

        spring = spec.add_tendon()
        spring.name = f"radial_spring_bc_{i}"
        spring.stiffness = [k_ex_phys, 0, 0]
        spring.damping = [damping, 0, 0]
        spring.rgba        = [0.9, 0.1, 0.1, 0.6]
        spring.width       = d
        spring.wrap_site(f"piv_bc_{i}_c")
        spring.wrap_site(f"piv_cb_{i}_c")
        # Compute rest length as distance between the two pivot sites at assembly
        rest_len = float(np.linalg.norm(
            (np.array(body_b.pos) + v_bc) - (np.array(body_c.pos) + v_cb)
        ))
        spring.springlength = [rest_len, rest_len]

    spec.option.timestep = 1e-5  # drop 20x; tune back up once stable
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    model = spec.compile()
    data = mujoco.MjData(model)



    mujoco.viewer.launch(model, data)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_obs(model, data, name="mod"):
    """Return [x, y, theta_z] of the top platform C in world frame."""
    sid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_obs")
    pos  = data.site_xpos[sid].copy()
    rot  = data.site_xmat[sid].reshape(3, 3)
    th_z = float(np.arctan2(rot[1, 0], rot[0, 0]))
    return np.array([pos[0], pos[1], th_z])


def apply_wrench(model, data, wrench, name="mod"):
    """Apply [Fx, Fy, Fz, Mx, My, Mz] to body C."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_C_site")
    bid = model.site_bodyid[sid]
    data.xfrc_applied[bid] = wrench


def simulate(model, data, wrench, duration=0.5, record_dt=0.01, name="mod"):
    """
    Apply a constant wrench to body C and record observations over time.

    Args:
        model:     Compiled MjModel.
        data:      MjData instance (will be reset).
        wrench:    [Fx, Fy, Fz, Mx, My, Mz] applied to body C [N, Nm].
        duration:  Simulation duration [s].
        record_dt: Observation recording interval [s].
        name:      Module name prefix.

    Returns:
        Dictionary with keys:
          't'   : (T,)   time array [s]
          'obs' : (T, 3) [x, y, theta_z] of platform C
    """
    mujoco.mj_resetData(model, data)
    steps        = int(duration / model.opt.timestep)
    record_every = max(1, int(record_dt / model.opt.timestep))
    times, obs   = [], []

    for step in range(steps):
        apply_wrench(model, data, wrench, name)
        mujoco.mj_step(model, data)
        if step % record_every == 0:
            times.append(data.time)
            obs.append(get_obs(model, data, name))

    return {"t": np.array(times), "obs": np.array(obs)}


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise(model, data, wrench=None, name="mod"):
    """
    Open the MuJoCo passive viewer and run the simulation live.

    Args:
        model:  Compiled MjModel.
        data:   MjData instance (will be reset).
        wrench: Optional [Fx, Fy, Fz, Mx, My, Mz] held constant during
                display. Defaults to zero (free settling under gravity).
        name:   Module name prefix, used for wrench application.
    """
    import mujoco.viewer

    if wrench is None:
        wrench = np.zeros(6)

    mujoco.mj_resetData(model, data)


    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.cam.distance = 0.05   # 5cm back
        viewer.cam.elevation = -20
        while viewer.is_running():
            apply_wrench(model, data, wrench, name)
            mujoco.mj_step(model, data)
            viewer.sync()


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test + display
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()



    # p = FlexureParams(
    #     d=1e-3, L=5e-3, E=1e6, nu=0.45, rho=1200.,
    #     r=4e-3, h=10e-3, n=3, damping_ratio=0.5,
    # )
    # p.summary()
    #
    # print("\nBuilding...")
    # spec  = build_module(p, name="mod")
    # model = spec.compile()
    # data  = mujoco.MjData(model)
    # print(f"  nq={model.nq}  nv={model.nv}  nbody={model.nbody}  njnt={model.njnt}")
    #
    # mujoco.mj_forward(model, data)
    # bid_B = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mod_B")
    # bid_C = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mod_C")
    # print(f"  B rest pos: {data.xpos[bid_B].round(4)}")
    # print(f"  C rest pos: {data.xpos[bid_C].round(4)}")
    #
    # cases = [
    #     ("Fz=-1N  (normal)",  np.array([ 0.,  0., -1., 0., 0., 0.    ])),
    #     ("Fx=+0.5N (shear)",  np.array([ 0.5, 0.,  0., 0., 0., 0.    ])),
    #     ("Fy=+0.5N (shear)",  np.array([ 0.,  0.5, 0., 0., 0., 0.    ])),
    #     ("Mz=+0.001Nm",       np.array([ 0.,  0.,  0., 0., 0., 0.001 ])),
    # ]
    #
    # print("\nSteady-state responses:")
    # for label, w in cases:
    #     res = simulate(model, data, w, duration=0.5, name="mod")
    #     o   = res["obs"]
    #     if np.any(np.isnan(o)) or np.any(np.abs(o) > 10.):
    #         print(f"  {label:24s}  DIVERGED")
    #     else:
    #         f = o[-1]
    #         print(f"  {label:24s}  "
    #               f"x={f[0]*1e3:+7.3f}mm  y={f[1]*1e3:+7.3f}mm  "
    #               f"θz={np.degrees(f[2]):+7.2f}°")
    #
    # print("\nLaunching viewer with Fx=0.5N shear (close window to exit)...")
    # visualise(model, data, wrench=np.array([0.5, 0., 0., 0., 0., 0.]), name="mod")