"""
Find a bezier curve which satisfies certain constraints.
Constraint list (26/02/09):
1. Spectrum of T(t) . vec{sigma} contains a null value.
2. Ratio of ||dT/dt|| / || i [dT/dt, T] || is fixed to sqrt((lambda**2 + 4) / (lambda**2 + 16))
3. Spectrum of dT/dt . vec{sigma} contains a null value. Should ideally be enforced through (1)
   automatically, but is not.

The constraints are enforced via a loss function minimization set up in optax.

The constraint that <0|H(t)|2> should be 0 can be enforced through `hamiltonian_loss()`, but is not
included by default in the program. Modify compute_loss and get_loss apprioprately in order to
include it.

Remarks:
* For some reason, the first iteration always gives the best loss values.
"""

import optax
import jax.numpy as jnp
import scipy.special as sp
from jax import jacfwd, jit
import jax

############################################
# Structure constant
############################################

sqrt3 = jnp.sqrt(3.0)
op_basis = jnp.array(
    [
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
        [[1 / sqrt3, 0, 0], [0, 1 / sqrt3, 0], [0, 0, -2 / sqrt3]],
    ],
    dtype=jnp.complex64,
)

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


lam = jnp.sqrt(2)

# Constraints corresponding to H_err = sigma_3, U(0) = Cos[th] sigma_1 + Sin[th] sigma_3 + diag(0,0,1):
# T(0) = Sin[2 th] e_1 - Cos[2 th] e_3
# N(0) = -2 e_2 - lambda * Cos[th] * e_5 + lambda * Sin[th] * e_7
# T(Tg) = - Sin[2 th] e_1 + Cos[2 th] e_3
# N(Tg) = 2 e_2 - lambda * Sin[th] * e_5 - lambda * Cos[th] * e_7
#
# The optimization parameters are marked in the list below. Use the same
# theta value and the optimized parameter values to generate the bezier curve from
# `bezier.py`.
theta = 0
init_w = jnp.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0],  # To set Omega(0) = 0
        [jnp.sin(2 * theta), 0, -jnp.cos(2 * theta), 0, 0, 0, 0, 0],  # To set T(0)
        [jnp.sin(2 * theta), 0, -jnp.cos(2 * theta), 0, 0, 0, 0, 0],  # To set T(0)
        [0, -2, 0, 0, -lam * jnp.cos(theta), 0, lam * jnp.sin(theta), 0],  # To set N(0)
        # Free parameters
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        # Constrained
        [0, -2, 0, 0, lam * jnp.sin(theta), 0, lam * jnp.cos(theta), 0],  # To set N(Tg)
        [-jnp.sin(2 * theta), 0, jnp.cos(2 * theta), 0, 0, 0, 0, 0],  # To set T(Tg)
        [-jnp.sin(2 * theta), 0, jnp.cos(2 * theta), 0, 0, 0, 0, 0],  # To set T(Tg)
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
)
n = init_w.shape[0] - 1
dim = init_w.shape[1]
bin_coeffs = jnp.array([sp.binom(n, i) for i in range(n + 1)])


@jit
def bezier_r(w, t):
    i = jnp.arange(n + 1)
    basis = bin_coeffs * (t**i) * ((1.0 - t) ** (n - i))  # shape (n+1,)
    return basis @ w  # (n+1,) @ (n+1, dim) → (dim,)


@jit
def bezier_T(w, t):
    dr = jacfwd(lambda tau: bezier_r(w, tau))(t)
    return dr / jnp.linalg.norm(dr)


@jit
def bezier_dT(w, t):
    return jit(jacfwd(lambda tau: bezier_T(w, tau)))(t)


@jit
def logm(b):
    res = jnp.zeros_like(b)
    ITERATIONS = 15
    for k in range(1, ITERATIONS):
        res += pow(-1, k + 1) * jnp.linalg.matrix_power(b - id, k) / k
    return res


def params_to_w(params):
    w = init_w.copy()
    w = w.at[4 : n - 3].set(params)
    return w


@jit
def create_operator(x):
    # x is shape (8,), op_basis is (8,3,3); want sum_i x[i] * op_basis[i]
    return jnp.tensordot(x, op_basis, axes=([0], [0]))


target = jnp.sqrt((lam**2 + 4) / (lam**2 + 16))
id = jnp.eye(3)


@jit
def ratio_loss(tangent, dtangent):
    """
    Ratio of ||dT/dt|| / || i [dT/dt, T] || should equal sqrt((lambda**2 + 4) / (lambda**2 + 16))
    """
    comm_t_dt = jax.vmap(comm, in_axes=(0, 0))(tangent, dtangent)
    dt_norm = jnp.linalg.norm(dtangent, axis=1)
    comm_norm = jnp.linalg.norm(comm_t_dt, axis=1)
    return jnp.mean(optax.l2_loss(dt_norm**2, (target * comm_norm) ** 2)) / 100


@jit
def dtangent_spectrum_loss(dtangent):
    """
    Spectrum of dT/dt should include 0. Should be automatically enforced if the spectrum of
    T(t) always contains a null value.
    """
    ops = jax.vmap(create_operator)(dtangent.astype(jnp.complex64))
    dets = jax.vmap(jnp.linalg.det)(ops)
    return jnp.mean(jnp.real(dets) ** 2) / 100


@jit
def tangent_spectrum_loss(tangent):
    """
    Spectrum of T(t) should include 0.
    """
    ops = jax.vmap(create_operator)(tangent.astype(jnp.complex64))
    dets = jax.vmap(jnp.linalg.det)(ops)
    return jnp.sum(jnp.real(dets) ** 2)


# def hamiltonian_loss(tangent):
#     """
#     <0|H(t)|2> should be 0.
#     """
#     ops = jnp.einsum("ijk,ni->njk", op_basis, tangent)
#     vals_, vecs = jnp.linalg.eigh(ops)
#     perm = jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
#     unitaries = jnp.transpose(vecs.conj() @ perm, (0, 2, 1))
#     hamiltonians = jnp.empty((n - 1, 3, 3), dtype=jnp.complex64)
#     error = jnp.empty((n - 1, 2))
#     for i in range(1, n):
#         # U(t) should be a smooth function. There can be discontinuities
#         # because of the diagonalization process: the eigenvectors can have
#         # arbitrary phases. Whenever a discontinuity is encountered, fix it
#         # it by adjusting the phase.
#         if jnp.sum(jnp.abs(unitaries[i] - unitaries[i - 1])) > 1e-3:
#             for j in range(3):
#                 if jnp.sum(jnp.abs(unitaries[i, j] - unitaries[i - 1, j])) > 1e-3:
#                     phases = jnp.angle(unitaries[i, j] / unitaries[i - 1, j])
#                     valid = jnp.logical_and(~jnp.isnan(phases), jnp.abs(phases) > 1e-8)
#                     idx = jnp.argmax(valid)
#                     phase = jnp.where(jnp.any(valid), phases[idx], 0.0)

#                     # phase = next(
#                     #     (ph for ph in phases if not jnp.isnan(ph) and abs(ph) > 1e-8), 0
#                     # )
#                     unitaries = unitaries.at[i, j].set(
#                         jnp.exp(-1j * phase) * unitaries[i, j]
#                     )
#         mat = 1j * logm(unitaries[i] @ unitaries[i - 1].conj().T) / 1
#         mat = 0.5 * (mat + mat.conj().T)
#         hamiltonians = hamiltonians.at[i - 1].set(mat)
#         error = error.at[i, 0].set(jnp.abs(jnp.trace(mat @ op_basis[3])))
#         error = error.at[i, 1].set(jnp.abs(jnp.trace(mat @ op_basis[4])))
#     return jnp.sum(jnp.sum(error, axis=0))


def get_loss(params, ts):
    w = params_to_w(params)
    dtangent = jax.vmap(bezier_dT, in_axes=(None, 0))(w, ts)
    tangent = jax.vmap(bezier_T, in_axes=(None, 0))(w, ts)
    return (
        tangent_spectrum_loss(tangent),
        dtangent_spectrum_loss(dtangent),
        ratio_loss(tangent, dtangent),
    )


def compute_loss(params, ts):
    tan_loss, dtan_loss, ratio_loss = get_loss(params, ts)
    return 0.1 * dtan_loss + ratio_loss + 20 * tan_loss


def array_to_str(a):
    if a.ndim == 0:
        return f"{a.item():.4f}"
    elif a.ndim == 1:
        return "[" + ", ".join(f"{jnp.abs(x):.4f}" for x in a) + "]\n"
    else:
        return "[" + ", ".join(array_to_str(x) for x in a) + "]"


def main():
    jnp.set_printoptions(precision=2)
    ts = jnp.linspace(0, 1, 1000)[1:-1]

    schedule = optax.exponential_decay(
        init_value=1e-2,  # initial learning rate
        transition_steps=10_000,  # steps before decay
        decay_rate=0.3,  # multiply by this decay factor
        staircase=True,  # decay in discrete intervals
    )
    optimizer = optax.adam(learning_rate=schedule)

    # Initialize parameters
    key = jax.random.PRNGKey(42)
    opt_sol = jnp.empty((n - 7, dim))
    min_loss = jnp.inf

    line_styles = ["-", "--", "-.", ":"]
    for repeats in range(1):
        key, subkey = jax.random.split(key)
        params = jax.random.normal(subkey, opt_sol.shape)
        opt_state = optimizer.init(params)

        grad = jit(jax.grad(lambda p: compute_loss(p, ts)))

        (
            dtan_spectra_loss_history,
            ratio_loss_history,
            tan_spectra_loss_history,
            ham_loss_history,
        ) = ([], [], [], [])

        print(f"Starting optimization #{repeats}")
        for idx in range(30_000):
            loss = [x.item() for x in get_loss(params, ts)]

            tan_spec_loss = loss[0]
            dtan_spec_loss = loss[1]
            ratio_loss = loss[2]

            tan_spectra_loss_history.append(tan_spec_loss)
            dtan_spectra_loss_history.append(dtan_spec_loss)
            ratio_loss_history.append(ratio_loss)
            # ham_loss_history.append(ham_loss)

            if tan_spec_loss < min_loss:
                min_loss = tan_spec_loss
                opt_sol = params
            grads = grad(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            if idx % 1000 == 0:
                print(f"Completed {idx} iterations. Last loss:", loss)

        print(f"Completed optimization #{repeats}")
        print("Final loss values:", loss)
        print("Parameters")
        with jnp.printoptions(precision=4):
            print(array_to_str(params))
        print(f"Min eig spread for T: ({min_loss}). Corresponding params:")
        with jnp.printoptions(precision=4):
            print(array_to_str(opt_sol))

    plt.clf()
    plt.grid()
    plt.plot(
        dtan_spectra_loss_history,
        c="b",
        linestyle=line_styles[repeats % len(line_styles)],
        label=f"Eig {repeats}",
    )
    plt.plot(
        ratio_loss_history,
        c="g",
        linestyle=line_styles[repeats % len(line_styles)],
        label=f"Ratio {repeats}",
    )
    plt.plot(
        tan_spectra_loss_history,
        c="r",
        linestyle=line_styles[repeats % len(line_styles)],
        label=f"Tangent {repeats}",
    )
    plt.plot(
        ham_loss_history,
        c="y",
        linestyle=line_styles[repeats % len(line_styles)],
        label=f"Tangent {repeats}",
    )
    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main()
