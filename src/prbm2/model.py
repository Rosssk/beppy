# pyright: strict

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray

# ---------------------------------------------------------------------------
# PRBM (Pseudo-Rigid-Body Model) constants
#   GAMMA       – ratio of the equivalent spring length to the true flexure
#                 length; sets the natural rest-length of each spring.
#   KAPPA_THETA – stiffness coefficient used by the caller to derive the
#                 rotational spring constant K_θ = KAPPA_THETA · E·I / L.
# ---------------------------------------------------------------------------
GAMMA: float = 0.85
KAPPA_THETA: float = 2.65

Vec3 = NDArray[np.float64]
Mat3 = NDArray[np.float64]
OptArray = ArrayLike | None


def as_vec3(v: ArrayLike) -> Vec3:
    """Convert *v* to an immutable (3,) float64 array."""
    arr: Vec3 = np.array(v, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"Expected 3D vector, got shape {arr.shape}")
    arr.flags.writeable = False
    return arr


def as_mat3(m: ArrayLike) -> Mat3:
    """Convert *m* to an immutable (3, 3) float64 array."""
    arr: Mat3 = np.array(m, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {arr.shape}")
    arr.flags.writeable = False
    return arr


def make_rotmat(angles: Vec3) -> Mat3:
    """Return the ZYX rotation matrix for the given Euler angles (radians).

    The convention is extrinsic: R = Rx · Ry · Rz, so the axes rotate in the
    order Z → Y → X when applied to column vectors.
    """
    rx_val, ry_val, rz_val = map(float, angles)

    rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx_val), -np.sin(rx_val)],
        [0, np.sin(rx_val), np.cos(rx_val)],
    ])
    ry = np.array([
        [np.cos(ry_val), 0, np.sin(ry_val)],
        [0, 1, 0],
        [-np.sin(ry_val), 0, np.cos(ry_val)],
    ])
    rz = np.array([
        [np.cos(rz_val), -np.sin(rz_val), 0],
        [np.sin(rz_val), np.cos(rz_val), 0],
        [0, 0, 1],
    ])
    return as_mat3(rx @ ry @ rz)


@dataclass(frozen=True, slots=True)
class ForceDef:
    """A concentrated force applied to a named body.

    Attributes:
        body:         Name of the body the force acts on.
        force:        Force vector in the global frame (units: N).
        attach_local: Point of application in the body's local frame.
    """
    body: str
    force: Vec3
    attach_local: Vec3


@dataclass(frozen=True, slots=True)
class BodyDef:
    """Rigid body with a pose (position + orientation).

    ``rotmat`` is derived from ``angles`` and stored for efficiency; it is
    computed in ``__post_init__`` so it is always consistent with the angles.

    Attributes:
        name:     Unique identifier used to reference this body.
        position: Origin of the body frame in global coordinates.
        angles:   Euler angles (rx, ry, rz) in radians.
        rotmat:   Rotation matrix corresponding to *angles* (read-only cache).
    """
    name: str
    position: Vec3
    angles: Vec3
    rotmat: Mat3

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotmat", make_rotmat(self.angles))

    @classmethod
    def create(cls, name: str, position: OptArray = None, angles: OptArray = None) -> "BodyDef":
        """Construct a :class:`BodyDef`, defaulting position and angles to zero."""
        _angles = as_vec3(np.zeros(3) if angles is None else angles)
        _position = as_vec3(np.zeros(3) if position is None else position)
        # rotmat is a required field but __post_init__ will overwrite it, so
        # we pass a placeholder to satisfy the dataclass constructor.
        return cls(name, _position, _angles, as_mat3(np.eye(3)))


@dataclass(frozen=True, slots=True)
class FlexureDef:
    """A compliant flexure modelled as a PRBM spring between two bodies.

    The spring is parameterised in each body's local frame so that it remains
    valid after the bodies are moved (prestressed configuration).

    Attributes:
        body_a / body_b:           Names of the connected bodies.
        attach_a_local / _b_local: Attachment points in respective local frames.
        spring_l0:                 Natural (unstressed) flexure length.
        spring_dir_a_local:        Unit vector from *a*'s attachment toward *b*,
                                   expressed in *a*'s local frame at creation.
        spring_dir_b_local:        Unit vector from *b*'s attachment toward *a*,
                                   expressed in *b*'s local frame at creation
    """
    body_a: str
    body_b: str
    attach_a_local: Vec3
    attach_b_local: Vec3

    spring_l0: float
    spring_dir_a_local: Vec3
    spring_dir_b_local: Vec3

    @classmethod
    def create(cls, body_a: BodyDef, body_b: BodyDef, attach_a_local: ArrayLike, attach_b_local: ArrayLike,
               spring_l0: float | None = None) -> FlexureDef:
        """Build a :class:`FlexureDef` from two bodies and their local attachment points.

        The spring direction vectors are computed from the bodies' current poses.
        If a body is later moved (prestressed), these vectors stay fixed — the
        deviation from the natural direction encodes the prestress.

        Args:
            body_a / body_b:          Bodies to connect.
            attach_a_local / _b_local: Attachment coordinates in local frames.
            spring_l0:                Natural length override; defaults to
                                      ``GAMMA * initial_distance``.
        """
        _attach_a = as_vec3(attach_a_local)
        _attach_b = as_vec3(attach_b_local)
        attach_a_global = body_a.position + body_a.rotmat @ _attach_a
        attach_b_global = body_b.position + body_b.rotmat @ _attach_b

        flexure_vec = attach_b_global - attach_a_global
        length_0 = float(np.linalg.norm(flexure_vec))

        if length_0 < 1e-12:
            raise ValueError(f"Attachment points of flexure between '{body_a.name}' and '{body_b.name}' coincide")

        # Unit vector pointing from a → b in the global frame.
        spring_unit_vec = flexure_vec / length_0
        spring_dir_a_local = as_vec3(body_a.rotmat.T @ spring_unit_vec)
        spring_dir_b_local = as_vec3(body_b.rotmat.T @ -spring_unit_vec)

        if spring_l0 is None:
            spring_l0 = length_0

        return cls(body_a.name, body_b.name, _attach_a, _attach_b, spring_l0, spring_dir_a_local, spring_dir_b_local)


class Model:
    """Container for bodies, flexures, and applied forces.

    Bodies are addressed by name. Flexures and forces keep string references
    so that moving a body (which replaces its :class:`BodyDef`) does not
    invalidate existing flexures — the changed geometry manifests as prestress.
    """

    def __init__(self) -> None:
        self.bodies: dict[str, BodyDef] = {}
        self.flexures: list[FlexureDef] = []
        self.forces: list[ForceDef] = []

    def _get_body(self, name: str) -> BodyDef:
        try:
            return self.bodies[name]
        except KeyError:
            raise KeyError(f"No body named {name!r}. Existing bodies: {list(self.bodies)}") from None

    def add_body(self, name: str, position: OptArray = None) -> None:
        """Add a new body at *position* (defaults to the origin)."""
        if name in self.bodies:
            raise ValueError(f"A body named {name!r} already exists")
        self.bodies[name] = BodyDef.create(name, position)

    def move_body(self, name: str, position: ArrayLike) -> None:
        """Translate *name* to a new *position*.

        Any flexures that reference this body are **not** updated; their spring
        directions remain fixed in each body's local frame. The displacement
        therefore introduces prestress into those flexures.
        """
        body = self._get_body(name)
        self.bodies[name] = BodyDef.create(name, position, body.angles)

    def rotate_body(self, name: str, angles: ArrayLike) -> None:
        """Reorient *name* to new Euler *angles* (radians).

        Same prestress semantics as :meth:`move_body`: existing flexure
        directions are frozen and the rotation manifests as prestress.
        """
        body = self._get_body(name)
        self.bodies[name] = BodyDef.create(name, body.position, angles)

    def add_flexure(self, body_a: str, attach_a_local: ArrayLike, body_b: str, attach_b_local: ArrayLike,
                    spring_l0: float | None = None) -> None:
        """Connect *body_a* and *body_b* with a PRBM spring flexure.

        Args:
            body_a / body_b:           Names of the bodies to connect.
            attach_a_local / _b_local: Local-frame attachment coordinates.
            spring_l0:                 Override for the natural spring length;
                                       defaults to ``GAMMA * current_distance``.
        """
        self.flexures.append(
            FlexureDef.create(
                self._get_body(body_a),
                self._get_body(body_b),
                attach_a_local,
                attach_b_local,
                spring_l0,
            ))

    def add_force(self, body: str, vector: ArrayLike, attach_local: OptArray = None) -> None:
        """Apply a concentrated force to *body*.

        Args:
            body:         Target body name.
            vector:       Force vector in the global frame (N).
            attach_local: Point of application in the body's local frame;
                          defaults to the body origin.
        """
        body_ = self._get_body(body)
        _attach = as_vec3(np.zeros(3) if attach_local is None else attach_local)
        self.forces.append(ForceDef(body_.name, as_vec3(vector), _attach))
