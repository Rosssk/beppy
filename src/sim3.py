"""
Pseudo-Rigid Body Model (PRBM) with RPR links
Structure: grounded body A -> 3 slanted beams -> body B -> 3 slanted beams -> body C

RPR link type:
  Rigid link - Rotational spring - Prismatic link (sliding DOF) - Rotational spring - Rigid link

Each flexure is modeled as: RPR
  - rigid segment from attachment point to PRB pivot
  - torsional spring at each pivot
  - prismatic (extensional) compliance along beam axis
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ─────────────────────────────────────────────────────────────────────────────
# Symbolic parameters
# ─────────────────────────────────────────────────────────────────────────────

h, r, n_sym = sp.symbols('h r n', positive=True)
k_r, k_p   = sp.symbols('k_r k_p', positive=True)   # rotational / prismatic stiffness
gamma       = sp.Symbol('gamma', real=True)            # PRB characteristic radius ratio (≈ 0.85)

# Generalised coordinates for each flexure (symbolic)
# We'll define them for flexure index i: θ1_i, δ_i, θ2_i
# (rotation at base pivot, prismatic extension, rotation at tip pivot)


# ─────────────────────────────────────────────────────────────────────────────
# RPR link kinematics helper
# ─────────────────────────────────────────────────────────────────────────────

def rotation_matrix_z(theta):
    """2-D rotation about Z, embedded in 3-D (acts in XY plane)."""
    c, s = sp.cos(theta), sp.sin(theta)
    return sp.Matrix([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def rotation_matrix_axis(axis_vec, theta):
    """Rodrigues rotation matrix about an arbitrary unit axis."""
    ax = sp.Matrix(axis_vec).normalized()
    ux, uy, uz = ax
    c, s = sp.cos(theta), sp.sin(theta)
    t = 1 - c
    return sp.Matrix([
        [t*ux*ux + c,    t*ux*uy - s*uz, t*ux*uz + s*uy],
        [t*ux*uy + s*uz, t*uy*uy + c,    t*uy*uz - s*ux],
        [t*ux*uz - s*uy, t*uy*uz + s*ux, t*uz*uz + c   ]
    ])


class RPRLink:
    """
    Symbolic RPR pseudo-rigid-body link.

    Geometry
    --------
    p_start : 3-vector (sympy)  – attachment on the "from" body
    p_end   : 3-vector (sympy)  – attachment on the "to"   body
    index   : int               – flexure index (used for unique symbol names)

    The beam natural axis  ê = (p_end – p_start) / |p_end – p_start|.
    The rigid portions have length  a = (1 – γ)/2 · L   (each side, symmetric RPR).
    The prismatic slide lives along ê with rest length  γ · L.

    Generalised coordinates
    -----------------------
    θ1  – rotation at the base  torsional spring (about the axis ⊥ ê in the bending plane)
    δ   – prismatic displacement along ê
    θ2  – rotation at the tip   torsional spring
    """

    def __init__(self, p_start, p_end, index, gamma_val=sp.Rational(85, 100)):
        self.p_start = sp.Matrix(p_start)
        self.p_end   = sp.Matrix(p_end)
        self.idx     = index
        self.gamma   = gamma_val

        # beam vector & length
        diff   = self.p_end - self.p_start
        self.L = sp.sqrt(diff.dot(diff))
        self.e_hat = diff / self.L          # unit beam axis

        # unique generalised coordinates
        self.theta1 = sp.Symbol(f'theta1_{index}', real=True)
        self.delta  = sp.Symbol(f'delta_{index}',  real=True)
        self.theta2 = sp.Symbol(f'theta2_{index}', real=True)

        # segment lengths
        self.a = (1 - self.gamma) / 2 * self.L   # each rigid arm
        self.b = self.gamma * self.L              # rest length of prismatic segment

        self._build_kinematics()
        self._build_potential_energy()

    # ------------------------------------------------------------------
    def _bending_axis(self):
        """
        Return a unit vector perpendicular to e_hat for the bending rotation.
        We pick the axis as e_hat × Z̃  (or e_hat × X̃ if nearly vertical).
        """
        Z = sp.Matrix([0, 0, 1])
        X = sp.Matrix([1, 0, 0])
        cross_Z = self.e_hat.cross(Z)
        # use X if e_hat is nearly parallel to Z
        mag_Z = sp.sqrt(cross_Z.dot(cross_Z))
        # (for symbolic, we keep both branches and let user substitute numbers)
        cross_X = self.e_hat.cross(X)
        # heuristic: return cross_Z (the caller can simplify after subs)
        return cross_Z / sp.sqrt(cross_Z.dot(cross_Z) + sp.Rational(1, 10**12))

    def _build_kinematics(self):
        """
        Forward kinematics: position of the tip given q = (θ1, δ, θ2).

        Frame walk:
          P0  = p_start
          P1  = P0 + R(θ1, b_axis) · (a · e_hat)        [end of base rigid arm]
          P2  = P1 + R(θ1, b_axis) · ((b + δ) · e_hat)  [end of prismatic + slide]
          P3  = P2 + R(θ1+θ2, b_axis) · (a · e_hat)     [end of tip rigid arm  = p_tip]

        For small deflections the bending axis is approximately fixed; we use
        the undeformed beam axis for the rotation axis (suitable for moderate deflections).
        """
        b_ax = self._bending_axis()

        R1 = rotation_matrix_axis(b_ax, self.theta1)
        R2 = rotation_matrix_axis(b_ax, self.theta1 + self.theta2)

        e  = self.e_hat
        P0 = self.p_start
        P1 = P0 + R1 * (self.a * e)
        P2 = P1 + R1 * ((self.b + self.delta) * e)
        P3 = P2 + R2 * (self.a * e)

        self.P0 = P0
        self.P1 = P1   # base  pivot
        self.P2 = P2   # tip   pivot  (also end of prismatic)
        self.P3 = P3   # tip attachment point

        # Tip position error (should equal p_end when q=0, used for verification)
        self.tip_residual = sp.simplify(P3.subs([(self.theta1, 0),
                                                  (self.delta,  0),
                                                  (self.theta2, 0)]) - self.p_end)

    def _build_potential_energy(self):
        """
        Strain energy:
          U = ½ k_r θ1²  +  ½ k_p δ²  +  ½ k_r θ2²
        """
        self.U = sp.Rational(1, 2) * k_r * self.theta1**2 \
               + sp.Rational(1, 2) * k_p * self.delta**2  \
               + sp.Rational(1, 2) * k_r * self.theta2**2

    @property
    def coordinates(self):
        return [self.theta1, self.delta, self.theta2]

    def __repr__(self):
        return (f"RPRLink({self.idx}: "
                f"p_start={self.p_start.T}, p_end={self.p_end.T}, "
                f"L={self.L})")


# ─────────────────────────────────────────────────────────────────────────────
# PRBM assembly
# ─────────────────────────────────────────────────────────────────────────────

class PRBM:
    """
    Pseudo-Rigid Body Model assembly.

    Bodies are rigid platforms described by a reference point (symbolic 3-vector).
    Flexures are RPR links connecting attachment points on two bodies.
    """

    def __init__(self, n_beams: int = 3):
        self.n = n_beams
        self.bodies   = {}   # name -> position vector (sympy Matrix)
        self.flexures = []   # list of RPRLink
        self._flexure_meta = []  # (body_from, attach_from, body_to, attach_to)

    def add_body(self, name: str, pos):
        self.bodies[name] = sp.Matrix(pos)

    def add_flexure(self, body_from: str, attach_from,
                          body_to:   str, attach_to):
        """
        attach_from / attach_to are positions expressed in global frame
        (as given in the pseudocode – already absolute coordinates).
        """
        idx  = len(self.flexures)
        link = RPRLink(attach_from, attach_to, idx)
        self.flexures.append(link)
        self._flexure_meta.append((body_from, sp.Matrix(attach_from),
                                    body_to,   sp.Matrix(attach_to)))

    # ------------------------------------------------------------------
    def total_potential_energy(self):
        return sp.Add(*[f.U for f in self.flexures])

    def all_coordinates(self):
        coords = []
        for f in self.flexures:
            coords.extend(f.coordinates)
        return coords

    def equations_of_motion(self, external_loads=None):
        """
        Returns the stiffness equations  ∂U/∂q = F  for each generalised coord.
        external_loads: dict {symbol: value} (optional)
        """
        U  = self.total_potential_energy()
        qs = self.all_coordinates()
        eqs = {}
        for q in qs:
            dU = sp.diff(U, q)
            F  = 0 if external_loads is None else external_loads.get(q, 0)
            eqs[q] = sp.Eq(dU, F)
        return eqs

    def summary(self):
        print("=" * 60)
        print(f"PRBM  –  {self.n} beams per layer,  {len(self.flexures)} flexures total")
        print("=" * 60)
        print("\nBodies:")
        for name, pos in self.bodies.items():
            print(f"  {name}: {pos.T}")
        print(f"\nFlexures (RPR links):")
        for i, (f, meta) in enumerate(zip(self.flexures, self._flexure_meta)):
            bf, af, bt, at = meta
            print(f"  [{i}] {bf}{list(af.T)} -> {bt}{list(at.T)}")
            print(f"       coords: θ1={f.theta1}, δ={f.delta}, θ2={f.theta2}")
            print(f"       L = {f.L}")
        print(f"\nTotal DOFs: {3 * len(self.flexures)}")
        U = self.total_potential_energy()
        print(f"\nTotal potential energy U:")
        print(f"  {U}")


# ─────────────────────────────────────────────────────────────────────────────
# Instantiate the model (using concrete symbolic π, h, r)
# ─────────────────────────────────────────────────────────────────────────────

def init_rbm():
    p = PRBM(3)
    n = 3

    p.add_body('A', (0, 0, 0))
    p.add_body('B', (0, 0, h / 2))
    p.add_body('C', (0, 0, h))

    for i in range(n):
        # Lower layer: A -> B
        p.add_flexure(
            'A', (r * sp.cos(i / n * 2 * sp.pi),
                  r * sp.sin(i / n * 2 * sp.pi),
                  0),
            'B', (r * sp.cos((i + 1) / n * 2 * sp.pi),
                  r * sp.sin((i + 1) / n * 2 * sp.pi),
                  h / 2)
        )
        # Upper layer: C -> B
        p.add_flexure(
            'C', (r * sp.cos((i - sp.Rational(1, 2)) / n * 2 * sp.pi),
                  r * sp.sin((i - sp.Rational(1, 2)) / n * 2 * sp.pi),
                  h),
            'B', (r * sp.cos((i + sp.Rational(1, 2)) / n * 2 * sp.pi),
                  r * sp.sin((i + sp.Rational(1, 2)) / n * 2 * sp.pi),
                  h / 2)
        )

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Numerical evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def eval_subs(expr, h_val=100.0, r_val=30.0):
    """Substitute concrete numbers for h and r (mm)."""
    return float(expr.subs([(h, h_val), (r, r_val)]))


def numeric_beam_points(link: RPRLink,
                        q_vals: dict,
                        h_val=100.0, r_val=30.0):
    """
    Evaluate the four characteristic points of an RPR link numerically.
    q_vals: {theta1: val, delta: val, theta2: val}
    Returns list of (x, y, z) tuples: [P0, P1, P2, P3]
    """
    subs = [(h, h_val), (r, r_val)] + list(q_vals.items())
    pts  = []
    for P in [link.P0, link.P1, link.P2, link.P3]:
        pts.append(tuple(float(P[j].subs(subs)) for j in range(3)))
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'body':        '#2E4057',
    'lower_rigid': '#E63946',
    'lower_spring':'#FF9F1C',
    'upper_rigid': '#457B9D',
    'upper_spring':'#2EC4B6',
    'pivot':       '#F4A261',
    'neutral':     '#A8DADC',
}


def draw_rpr_beam(ax, pts, color_rigid, color_spring, lw=2.5, deformed=False):
    """Draw one RPR link: P0-P1 rigid | P1-P2 spring | P2-P3 rigid."""
    P0, P1, P2, P3 = [np.array(p) for p in pts]

    # rigid arms
    for A, B in [(P0, P1), (P2, P3)]:
        ax.plot(*zip(A, B), color=color_rigid, lw=lw, solid_capstyle='round')

    # prismatic+spring segment (dashed)
    ax.plot(*zip(P1, P2), color=color_spring, lw=lw,
            linestyle='--', dashes=(4, 2))

    # pivots
    for P in [P1, P2]:
        ax.scatter(*P, color=COLORS['pivot'], s=40, zorder=5)


def draw_body_disk(ax, center, radius=5.0, color=COLORS['body'], alpha=0.7):
    """Draw a thin disk representing a rigid body platform."""
    theta = np.linspace(0, 2 * np.pi, 60)
    cx, cy, cz = center
    xs = cx + radius * np.cos(theta)
    ys = cy + radius * np.sin(theta)
    zs = np.full_like(xs, cz)

    # filled polygon
    verts = [list(zip(xs, ys, zs))]
    poly  = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor='white', lw=0.8)
    ax.add_collection3d(poly)


def visualise(prbm: PRBM,
              q_vals: dict = None,
              h_val: float = 100.0,
              r_val: float = 30.0,
              title: str   = "PRBM – RPR flexure model"):
    """
    3-D visualisation of the PRBM in the reference (undeformed) or
    deformed configuration.

    q_vals: dict {symbol: float_value}  – omit for undeformed (all zeros)
    """
    if q_vals is None:
        q_vals = {}

    fig = plt.figure(figsize=(12, 9))

    # ── left: 3-D isometric ──────────────────────────────────────────────
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.set_title(title, fontsize=11, pad=10)

    disk_r = r_val * 1.3

    # body platforms
    for name, pos_sym in prbm.bodies.items():
        cx = float(pos_sym[0].subs([(h, h_val), (r, r_val)]))
        cy = float(pos_sym[1].subs([(h, h_val), (r, r_val)]))
        cz = float(pos_sym[2].subs([(h, h_val), (r, r_val)]))
        draw_body_disk(ax3d, (cx, cy, cz), radius=disk_r)
        ax3d.text(cx + disk_r * 0.6, cy + disk_r * 0.6, cz + 3,
                  name, fontsize=13, fontweight='bold', color=COLORS['body'])

    # beams
    n_fl = len(prbm.flexures)
    for idx, link in enumerate(prbm.flexures):
        q_link = {link.theta1: q_vals.get(link.theta1, 0.0),
                  link.delta:  q_vals.get(link.delta,  0.0),
                  link.theta2: q_vals.get(link.theta2, 0.0)}
        pts = numeric_beam_points(link, q_link, h_val, r_val)

        is_lower = idx < prbm.n     # first n flexures are lower layer
        c_rig = COLORS['lower_rigid']  if is_lower else COLORS['upper_rigid']
        c_spr = COLORS['lower_spring'] if is_lower else COLORS['upper_spring']
        draw_rpr_beam(ax3d, pts, c_rig, c_spr)

    # ground symbol at body A
    az = 0.0
    ax3d.scatter(0, 0, az - 3, marker='^', color='gray', s=120, zorder=6)
    ax3d.text(0, 0, az - 10, 'Ground', ha='center', color='gray', fontsize=8)

    lim = disk_r * 1.5
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-15, h_val + 15)
    ax3d.set_xlabel('X (mm)', labelpad=6)
    ax3d.set_ylabel('Y (mm)', labelpad=6)
    ax3d.set_zlabel('Z (mm)', labelpad=6)
    ax3d.view_init(elev=25, azim=40)

    # ── right: side-view schematic with RPR annotation ───────────────────
    ax2d = fig.add_subplot(122)
    ax2d.set_title("Side-view schematic with RPR link detail", fontsize=11)
    ax2d.set_aspect('equal')
    ax2d.axis('off')

    sw = 60       # schematic half-width
    sh = h_val    # schematic height

    # body bars
    for z_frac in [0, 0.5, 1.0]:
        z = z_frac * sh
        label = {0: 'Body A (grounded)', 0.5: 'Body B', 1.0: 'Body C'}[z_frac]
        ax2d.fill_between([-sw, sw], [z - 2, z - 2], [z + 2, z + 2],
                           color=COLORS['body'], alpha=0.85)
        ax2d.text(sw + 3, z, label, va='center', fontsize=9,
                  color=COLORS['body'], fontweight='bold')

    # schematic beams (projected, not to scale – illustrative)
    beam_xs_lower = [(-sw * 0.6, -sw * 0.05), (0, sw * 0.6), (sw * 0.05, -sw * 0.55)]
    beam_xs_upper = [(-sw * 0.55, sw * 0.05), (sw * 0.55, 0), (-sw * 0.05, sw * 0.55)]

    def draw_rpr_schematic(ax, x0, z0, x3, z3, c_rig, c_spr, gamma_val=0.85):
        arm = (1 - gamma_val) / 2
        x1 = x0 + arm * (x3 - x0);  z1 = z0 + arm * (z3 - z0)
        x2 = x0 + (1 - arm) * (x3 - x0); z2 = z0 + (1 - arm) * (z3 - z0)
        ax.plot([x0, x1], [z0, z1], color=c_rig, lw=2.5, solid_capstyle='round')
        ax.plot([x1, x2], [z1, z2], color=c_spr, lw=2.5, linestyle='--', dashes=(4, 2))
        ax.plot([x2, x3], [z2, z3], color=c_rig, lw=2.5, solid_capstyle='round')
        for xp, zp in [(x1, z1), (x2, z2)]:
            ax.scatter(xp, zp, color=COLORS['pivot'], s=45, zorder=5)

    for (x0, x3) in beam_xs_lower:
        draw_rpr_schematic(ax2d, x0, 0, x3, sh / 2,
                           COLORS['lower_rigid'], COLORS['lower_spring'])
    for (x0, x3) in beam_xs_upper:
        draw_rpr_schematic(ax2d, x0, sh, x3, sh / 2,
                           COLORS['upper_rigid'], COLORS['upper_spring'])

    # RPR label with arrow spanning one beam
    x0_ex, z0_ex = -sw * 0.6, 0
    x3_ex, z3_ex = -sw * 0.05, sh / 2
    arm = 0.075
    x1_ex = x0_ex + arm * (x3_ex - x0_ex); z1_ex = z0_ex + arm * (z3_ex - z0_ex)
    x2_ex = x0_ex + (1 - arm) * (x3_ex - x0_ex); z2_ex = z0_ex + (1 - arm) * (z3_ex - z0_ex)

    offset = 6
    for (xa, za), lbl in [((x0_ex, z0_ex), 'R'), ((x1_ex, z1_ex), 'P'), ((x2_ex, z2_ex), 'R')]:
        ax2d.text(xa - offset, za, lbl, fontsize=10, color='#333',
                  ha='right', va='center', fontstyle='italic')

    ax2d.set_xlim(-sw - 20, sw + 40)
    ax2d.set_ylim(-15, sh + 15)

    # legend
    legend_items = [
        mpatches.Patch(color=COLORS['lower_rigid'],  label='Rigid arm (lower layer)'),
        mpatches.Patch(color=COLORS['lower_spring'], label='PRB spring segment (lower)'),
        mpatches.Patch(color=COLORS['upper_rigid'],  label='Rigid arm (upper layer)'),
        mpatches.Patch(color=COLORS['upper_spring'], label='PRB spring segment (upper)'),
        mpatches.Patch(color=COLORS['pivot'],        label='Torsional + prismatic pivot'),
        mpatches.Patch(color=COLORS['body'],         label='Rigid body platform'),
    ]
    fig.legend(handles=legend_items, loc='lower center',
               ncol=3, fontsize=8, frameon=True,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Build the model
    prbm = init_rbm()
    prbm.summary()

    # Print EOM (first few, unloaded)
    print("\n── Equations of motion (unloaded, first 3 coords) ──")
    eoms = prbm.equations_of_motion()
    for q, eq in list(eoms.items())[:3]:
        print(f"  ∂U/∂{q} = 0  →  {eq}")

    # Verify tip residual (should be zero vector at q=0)
    print("\n── Tip residual check (should be [0,0,0]) ──")
    for i, link in enumerate(prbm.flexures[:2]):
        res = [sp.simplify(x) for x in link.tip_residual]
        print(f"  Flexure {i}: {res}")

    # ── Visualise undeformed configuration ─────────────────────────────
    fig_ref = visualise(prbm, h_val=100, r_val=30,
                        title="PRBM – undeformed (reference) configuration")

    # ── Visualise a small deformed configuration ────────────────────────
    # Apply a small twist: give every flexure a small θ1 rotation
    q_deformed = {}
    small_angle = 0.08   # rad
    small_delta = 1.5    # mm
    for link in prbm.flexures:
        q_deformed[link.theta1] =  small_angle
        q_deformed[link.delta]  =  small_delta
        q_deformed[link.theta2] = -small_angle * 0.5

    fig_def = visualise(prbm, q_vals=q_deformed, h_val=100, r_val=30,
                        title="PRBM – deformed configuration (θ₁=0.08 rad, δ=1.5 mm)")

    plt.savefig('./prbm_rpr.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to /mnt/user-data/outputs/prbm_rpr.png")
    plt.show()