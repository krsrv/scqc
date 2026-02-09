"""
Calculate space curve properties corresponding to an explicity calculated form
of the tangent curve.
"""

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jit

# 2 dimensional Hilbert space
sparse_M = [
    (0, 2, 1, 2),
    (0, 1, 2, -2),
    (1, 2, 0, -2),
    (1, 0, 2, 2),
    (2, 1, 0, 2),
    (2, 0, 1, -2),
]


idx = jnp.array([(k, i, j) for k, i, j, v in sparse_M])
vals = jnp.array([v for _, _, _, v in sparse_M])

k_idx = idx[:, 0]
i_idx = idx[:, 1]
j_idx = idx[:, 2]


def cross(T1, T2):
    """Calculate T1 x T2"""
    contrib = vals * T1[i_idx] * T2[j_idx]
    return jnp.zeros_like(T1).at[k_idx].add(contrib)


dim = 3


def T(t):
    return jnp.array([0, jnp.sin(jnp.pi * t), jnp.cos(jnp.pi * t), 0, 0, 0, 0, 0])


d_T = jacfwd(T)
dd_T = jacfwd(d_T)


def N(t):
    unnorm_N = d_T(t)
    return unnorm_N / jnp.linalg.norm(unnorm_N)


T = jit(T)
d_T = jit(d_T)
dd_T = jit(dd_T)
N = jit(N)
cross = jit(cross)

TN = 1000
dt = 1.0 / TN
omega_list, torsion_list = [], []
normal_list = np.empty((TN + 1, dim**2 - 1), dtype=np.float32)
tangent_list = np.empty((TN + 1, dim**2 - 1), dtype=np.float32)

for i, t in enumerate(jnp.linspace(0, 1, TN + 1)):
    curvature = jnp.linalg.norm(d_T(t))

    # 2 level system
    omega = curvature
    torsion = 2 * dd_T(t) @ cross(T(t), d_T(t)) / (4 * omega**2)

    omega_list.append(omega.item())
    torsion_list.append(torsion.item())

    normal, tangent = N(t), T(t)
    normal_list[i] = normal
    tangent_list[i] = tangent

omega_list = np.array([0 if np.isnan(x) else x for x in omega_list])
torsion_list = np.array([0 if np.isnan(x) else x for x in torsion_list])
normal_list = np.where(np.isnan(normal_list), 0, normal_list)
tangent_list = np.where(np.isnan(tangent_list), 0, tangent_list)

np.savez(
    "curve.npz",
    omega=omega_list,
    torsion=torsion_list,
    normal=normal_list,
    tangent=tangent_list,
)

print("Boundary conditions:")
with jnp.printoptions(precision=2):
    print("  t = 0:")
    print("    T", T(0.01))
    print("    dT", d_T(0.01))
    print("    N", N(0.001))

    print("  t = 1:")
    print("    T", T(0.999))
    print("    dT", d_T(0.999))
    print("    N", N(0.999))
