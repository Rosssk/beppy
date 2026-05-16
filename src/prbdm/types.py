from jaxtyping import Array, Float
from typing import NamedTuple, TypeAlias

# Shapes are annotated as jaxtyping dimension strings.
Vec3 = Float[Array, "3"]                        # world/body 3-vector
Mat33: TypeAlias = Float[Array, "3 3"]          # rotation or skew matrix

# Mass matrix
MassMat: TypeAlias = Float[Array, "12 12"]

# Scalar (0-D JAX array)
Scalar: TypeAlias = Float[Array, ""]


# Identical types, just with different names to make code easier to read
class BodyState(NamedTuple):
    pos: Vec3   
    rot: Vec3   

class BodyStateDot(NamedTuple):
    vel: Vec3   
    rotvel: Vec3   

FullState: TypeAlias = tuple[BodyState, BodyState, BodyState]               # Body states [0-2]
FullStateDot: TypeAlias = tuple[BodyStateDot, BodyStateDot, BodyStateDot]   # Body state derivatives [0-2]
ReducedState: TypeAlias = tuple[BodyState, BodyState]                       # Body states [1-2]
ReducedStateDot: TypeAlias = tuple[BodyStateDot, BodyStateDot]              # Body state derivatives [0-2]
