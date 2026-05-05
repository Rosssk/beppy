from jax import Array
from jaxtyping import Float, Int






Vec3 = Float[Array, "3"]
Mat33 = Float[Array, "3 3"]

# Batched
VecN3 = Float[Array, "N 3"]
MatN33 = Float[Array, "N 3 3"]

IndexVec = Int[Array, "N"]
IndexMat = Int[Array, "N 2"]