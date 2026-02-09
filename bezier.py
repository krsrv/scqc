"""
Calculate space curve properties corresponding to a given list of bezier curve
parameters. The parameters are chosen based on:
1. Boundary conditions decided by the unitary we wish to implement
2. Restriction to the family of curves which would correspond to the drive
 Hamiltonian structure. (`find_constrained_curve.py`)
"""

import jax.numpy as jnp
import numpy as np
import scipy.special as sp
from jax import jacfwd, jit

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


############################################

lam = jnp.sqrt(2)
# Constraints corresponding to H_err = sigma_8
# T(0) = e_8, N(0) = e_7, T(Tg) = e_8, N(Tg) = e_5 are:
#   w_0 = w_{n} = 0
#   w_1 = w_2 = e_8
#   w_{n-1} = w_{n-2} = -e_8
#   w_3 = e_7
#   w_{n-3} = e_5

# Constraints corresponding to H_err = sigma_3, U(0) = Id
# T(0) = e_3, N(0) = 2 e_2 - lambda * e_7
# T(Tg) = -e_3, N(Tg) = -2 e_2 - lam * e_5 are:
#   w_0 = w_{n} = 0
#   w_1 = w_2 = e_3
#   w_{n-1} = w_{n-2} = e_3
#   w_3 = sqrt(2) e_2 - e_7
#   w_{n-3} = sqrt(2) e_2 + e_7

# Constraints corresponding to H_err = sigma_3, U(0) = Cos[th] sigma_1 + Sin[th] sigma_3 + diag(0,0,1):
# T(0) = Sin[2 th] e_1 - Cos[2 th] e_3
# N(0) = -2 e_2 - lambda * Cos[th] * e_5 + lambda * Sin[th] * e_7
# T(Tg) = - Sin[2 th] e_1 + Cos[2 th] e_3
# N(Tg) = 2 e_2 - lambda * Sin[th] * e_5 - lambda * Cos[th] * e_7

# 3 dimensional Hilbert space
theta = 0
w = jnp.array(
    [
        # Constrained
        [0, 0, 0, 0, 0, 0, 0, 0],  # To set Omega(0) = 0
        [jnp.sin(2 * theta), 0, -jnp.cos(2 * theta), 0, 0, 0, 0, 0],  # To set T(0)
        [jnp.sin(2 * theta), 0, -jnp.cos(2 * theta), 0, 0, 0, 0, 0],  # To set T(0)
        [0, -2, 0, 0, -lam * jnp.cos(theta), 0, lam * jnp.sin(theta), 0],  # To set N(0)
        # Free parameters
        # fmt: off
        # # 5 points, 4.8e-5 eigenvalue loss for T, theta=pi/8
        # [ 14.69,   4.94, -22.52,   5.04,   3.11,  -1.03,  -2.,     0.52,],
        # [ -3.84,  -3.76,  -8.48,  -1.64,   1.18,   0.92,   4.05,   0.46,],
        # [-12.11,  -3.86,  -5.11,  -2.68,   5.02,  -7.33,  -7.36,   0.18,],
        # [ -9.69,  -2.71,   1.51,  -3.7,    4.73,  -0.04,   4.,     0.31,],
        # [-19.85,   2.17,   5.56,  -3.77,   3.71,  -5.8,   -5.37,   0.61,],
        # # 5 points, 4.8e-5 eigenvalue loss for T, theta=0
        [ -5.56,   5.34, -26.76,   5.03,   3.68,   1.03,  -0.66,   0.49,],
        [ -9.09,  -4.46,  -5.05,  -1.51,  -0.16,  -0.12,   4.51,   0.59,],
        [-15.15,  -1.99,   4.4,    0.31,   8.14,  -7.75,  -5.18,   0.2, ],
        [ -6.01,  -4.51,  10.17,  -3.88,   3.49,  -2.72,   5.69,   0.41,],
        [ -9.22,   2.84,  17.71,  -0.99,   4.81,  -6.1,   -3.63,   0.58,],
        # fmt: on
        # Constrained
        [0, -2, 0, 0, lam * jnp.sin(theta), 0, lam * jnp.cos(theta), 0],  # To set N(Tg)
        [-jnp.sin(2 * theta), 0, jnp.cos(2 * theta), 0, 0, 0, 0, 0],  # To set T(Tg)
        [-jnp.sin(2 * theta), 0, jnp.cos(2 * theta), 0, 0, 0, 0, 0],  # To set T(Tg)
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
)
# # 2 dimensional Hilbert space
# w = jnp.array(
#     [
#         # Constrained
#         [0, 0, 0],
#         [0, 0, 1],
#         [0, 0, 1],
#         [0, 1, 0],
#         # Free parameters
#         [1, 0, 0],
#         [1, -1, 0],
#         [0, -1, 1],
#         # Constrained
#         [0, -1, 0],
#         [0, 0, 1],
#         [0, 0, 1],
#         [0, 0, 0],
#     ]
# )
n = w.shape[0] - 1
dim = 3 if w.shape[1] == 8 else 2
bin_coeffs = jnp.array([sp.binom(n, i) for i in range(n + 1)])


def r(t):
    """
    Bezier curve r(t)
    """
    i = jnp.arange(n + 1)
    basis = bin_coeffs * (t**i) * ((1.0 - t) ** (n - i))  # shape (n+1,)
    return basis @ w  # (n+1,) @ (n+1, dim) → (dim,)


d_r = jacfwd(r)


def T(t):
    i = jnp.arange(n + 1)
    basis = bin_coeffs * (
        i * (t ** (i - 1)) * ((1.0 - t) ** (n - i))
        + (t**i) * (n - i) * ((1.0 - t) ** (n - i - 1))
    )
    unnorm = basis @ w
    return unnorm / jnp.linalg.norm(unnorm)
    # unnorm_T = d_r(t)
    # return unnorm_T / jnp.linalg.norm(unnorm_T)


d_T = jacfwd(T)
dd_T = jacfwd(d_T)
ddd_T = jacfwd(dd_T)


def N(t):
    unnorm_N = d_T(t)
    return unnorm_N / jnp.linalg.norm(unnorm_N)


r = jit(r)
T = jit(T)
d_T = jit(d_T)
dd_T = jit(dd_T)
N = jit(N)
comm = jit(comm)


TN = 1000
dt = 1.0 / TN
omega_list, torsion_list, borsion_list = [], [], []
ratio_list = []
normal_list = np.empty((TN + 1, dim**2 - 1), dtype=np.float32)
tangent_list = np.empty((TN + 1, dim**2 - 1), dtype=np.float32)

for i, t in enumerate(jnp.linspace(0, 1, TN + 1)):
    curvature = jnp.linalg.norm(d_T(t))

    # 3 dimensional Hilbert space
    omega = 2 * curvature / jnp.sqrt(4 + lam**2)
    torsion = 4 * dd_T(t) @ comm(T(t), d_T(t)) / omega**2
    borsion = -4 * dd_T(t) @ comm(square(T(t)), d_T(t)) / omega**2

    ratio = jnp.linalg.norm(d_T(t)) / jnp.linalg.norm(comm(T(t), d_T(t)))

    # # 2 dimensional Hilbert space
    # omega = curvature
    # s, c = d_T(t)[0:2]
    # torsion = 2 * dd_T(t) @ comm(T(t), d_T(t)) / (-4 * omega**2)

    omega_list.append(omega.item())
    torsion_list.append(torsion.item())
    borsion_list.append(borsion.item())
    ratio_list.append(ratio.item())

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
    ratio=ratio_list,
)

print("Boundary conditions:")
with jnp.printoptions(precision=2):
    print("  t = 0:")
    print("    T", T(0.01))
    print("    dT", d_T(0.01))
    print("    ddT", ddd_T(0.01) / jnp.linalg.norm(ddd_T(0.01)))
    print("    N", N(0.001))

    print("  t = 1:")
    print("    T", T(0.999))
    print("    dT", d_T(0.999))
    print("    ddT", ddd_T(0.999) / jnp.linalg.norm(ddd_T(0.999)))
    print("    N", N(0.999))
