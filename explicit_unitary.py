"""
Calculate space curve properties corresponding to a unitary
generated from a time independent Hamiltonian.
"""

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jit
import jax

############################################
# Structure constant
############################################

sqrt3 = np.sqrt(3.0)

# list of (k, i, j, value)
# 3 dimensional Hilbert space
sparse_M = [
    # R[0]
    (0, 2, 1, 2),
    (0, 1, 2, -2),
    (0, 6, 3, 1),
    (0, 5, 4, -1),
    (0, 4, 5, 1),
    (0, 3, 6, -1),
    # R[1]
    (1, 2, 0, -2),
    (1, 0, 2, 2),
    (1, 5, 3, 1),
    (1, 6, 4, 1),
    (1, 3, 5, -1),
    (1, 4, 6, -1),
    # R[2]
    (2, 1, 0, 2),
    (2, 0, 1, -2),
    (2, 4, 3, 1),
    (2, 3, 4, -1),
    (2, 6, 5, -1),
    (2, 5, 6, 1),
    # R[3]
    (3, 6, 0, -1),
    (3, 5, 1, -1),
    (3, 4, 2, -1),
    (3, 2, 4, 1),
    (3, 7, 4, sqrt3),
    (3, 1, 5, 1),
    (3, 0, 6, 1),
    (3, 4, 7, -sqrt3),
    # R[4]
    (4, 5, 0, 1),
    (4, 6, 1, -1),
    (4, 3, 2, 1),
    (4, 2, 3, -1),
    (4, 7, 3, -sqrt3),
    (4, 0, 5, -1),
    (4, 1, 6, 1),
    (4, 3, 7, sqrt3),
    # R[5]
    (5, 4, 0, -1),
    (5, 3, 1, 1),
    (5, 6, 2, 1),
    (5, 1, 3, -1),
    (5, 0, 4, 1),
    (5, 2, 6, -1),
    (5, 7, 6, sqrt3),
    (5, 6, 7, -sqrt3),
    # R[6]
    (6, 3, 0, 1),
    (6, 4, 1, 1),
    (6, 5, 2, -1),
    (6, 0, 3, -1),
    (6, 1, 4, -1),
    (6, 2, 5, 1),
    (6, 7, 5, -sqrt3),
    (6, 5, 7, sqrt3),
    # R[7]
    (7, 4, 3, sqrt3),
    (7, 3, 4, -sqrt3),
    (7, 6, 5, sqrt3),
    (7, 5, 6, -sqrt3),
]

sparse_sq = [
    # R[0]
    (0, 3, 5, 1),
    (0, 4, 6, 1),
    (0, 0, 7, 2 / sqrt3),
    # R[1]
    (1, 4, 5, 1),
    (1, 3, 6, -1),
    (1, 1, 7, 2 / sqrt3),
    # R[2]
    (2, 3, 3, 1 / 2),
    (2, 4, 4, 1 / 2),
    (2, 5, 5, -1 / 2),
    (2, 6, 6, -1 / 2),
    (2, 2, 7, 2 / sqrt3),
    # R[3]
    (3, 2, 3, 1),
    (3, 0, 5, 1),
    (3, 1, 6, -1),
    (3, 3, 7, -1 / sqrt3),
    # R[4]
    (4, 2, 4, 1),
    (4, 1, 5, 1),
    (4, 0, 6, 1),
    (4, 4, 7, -1 / sqrt3),
    # R[5]
    (5, 0, 3, 1),
    (5, 1, 4, 1),
    (5, 2, 5, -1),
    (5, 5, 7, -1 / sqrt3),
    # R[6]
    (6, 1, 3, -1),
    (6, 0, 4, 1),
    (6, 2, 6, -1),
    (6, 6, 7, -1 / sqrt3),
    # R[7]
    (7, 0, 0, 1 / sqrt3),
    (7, 1, 1, 1 / sqrt3),
    (7, 2, 2, 1 / sqrt3),
    (7, 3, 3, -1 / (2 * sqrt3)),
    (7, 4, 4, -1 / (2 * sqrt3)),
    (7, 5, 5, -1 / (2 * sqrt3)),
    (7, 6, 6, -1 / (2 * sqrt3)),
    (7, 7, 7, -1 / sqrt3),
]
# # 2 dimensional Hilbert space
# sparse_M = [
#     (0, 2, 1, 2),
#     (0, 1, 2, -2),
#     (1, 2, 0, -2),
#     (1, 0, 2, 2),
#     (2, 1, 0, 2),
#     (2, 0, 1, -2),
# ]


comm_idx = jnp.array([(k, i, j) for k, i, j, v in sparse_M])
comm_vals = jnp.array([v for _, _, _, v in sparse_M])

k_comm_idx = comm_idx[:, 0]
i_comm_idx = comm_idx[:, 1]
j_comm_idx = comm_idx[:, 2]


def comm(T1, T2):
    """
    Compute T, where T.sigma = i[T1.sigma, T2.sigma].
    """
    contrib = comm_vals * T1[i_comm_idx] * T2[j_comm_idx]
    return jnp.zeros_like(T1).at[k_comm_idx].add(contrib)


sq_idx = jnp.array([(k, i, j) for k, i, j, v in sparse_sq])
sq_vals = jnp.array([v for _, _, _, v in sparse_sq])

k_sq_idx = sq_idx[:, 0]
i_sq_idx = sq_idx[:, 1]
j_sq_idx = sq_idx[:, 2]


def square(T1):
    contrib = sq_vals * T1[i_sq_idx] * T1[j_sq_idx]
    return jnp.zeros_like(T1).at[k_sq_idx].add(contrib)


basis = jnp.array(
    [
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, -2]] / np.sqrt(3),
    ],
    dtype=np.complex128,
)

############################################

dim = 3
ll = jnp.sqrt(2)


def T(t):
    """
    Compute tangent vector T(t) = U^\dagger sigma_3 U
    """
    U = jax.scipy.linalg.expm(-1j * jnp.pi / sqrt3 * t * (basis[1] + ll * basis[6]))
    operator = U.conj().T @ basis[2] @ U
    unnorm_T = jnp.real(jnp.trace(basis @ operator, axis1=-1, axis2=-2)) / 2
    return unnorm_T / jnp.linalg.norm(unnorm_T)


d_T = jacfwd(T)
dd_T = jacfwd(d_T)


def N(t):
    unnorm_N = d_T(t)
    return unnorm_N / jnp.linalg.norm(unnorm_N)


T = jit(T)
d_T = jit(d_T)
dd_T = jit(dd_T)
N = jit(N)
comm = jit(comm)


TN = 1000
dt = 1.0 / TN
omega_list, torsion_list, borsion_list = [], [], []
normal_list = np.empty((TN + 1, dim**2 - 1), dtype=np.float32)
tangent_list = np.empty((TN + 1, dim**2 - 1), dtype=np.float32)

for i, t in enumerate(jnp.linspace(0, 1, TN + 1)):
    curvature = jnp.linalg.norm(d_T(t))

    # 3 dimensional Hilbert space
    omega = 2 * curvature / jnp.sqrt(4 + ll**2)
    torsion = 4 * dd_T(t) @ comm(T(t), d_T(t)) / omega**2
    borsion = -4 * dd_T(t) @ comm(square(T(t)), d_T(t)) / omega**2

    # # 2 dimensional Hilbert space
    # omega = curvature
    # s, c = d_T(t)[0:2]
    # torsion = 2 * dd_T(t) @ comm(T(t), d_T(t)) / (-4 * omega**2)

    omega_list.append(omega.item())
    torsion_list.append(torsion.item())
    borsion_list.append(borsion.item())

    normal, tangent = N(t), T(t)
    normal_list[i] = normal
    tangent_list[i] = tangent

omega_list = np.array([0 if np.isnan(x) else x for x in omega_list])
torsion_list = np.array([0 if np.isnan(x) else x for x in torsion_list])
borsion_list = np.array([0 if np.isnan(x) else x for x in borsion_list])
normal_list = np.where(np.isnan(normal_list), 0, normal_list)
tangent_list = np.where(np.isnan(tangent_list), 0, tangent_list)


np.savez(
    "curve.npz",
    omega=omega_list,
    torsion=torsion_list,
    borsion=borsion_list,
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
