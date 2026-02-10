"""
Given tangent curve and other space curve properties (e.g. torsion) as functions of
time, extract the unitary.
"""

import numpy as np
from utils import Ops, calculate_unitary, normalize
from scipy.linalg import logm

basis = np.array(
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

tg = 1  # ns


def extract_pulse_v2(tangent: np.ndarray):
    """
    Use diagonalization to extract unitary
    """
    n = tangent.shape[0]

    ops = np.einsum("ijk,ni->njk", basis, tangent)
    vals_, vecs = np.linalg.eigh(ops)
    perm = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    unitaries = np.transpose(vecs.conj() @ perm, (0, 2, 1))
    hamiltonians = np.empty((n - 1, 3, 3), dtype=np.complex128)
    for i in range(1, n):
        # U(t) should be a smooth function. There can be discontinuities
        # because of the diagonalization process: the eigenvectors can have
        # arbitrary phases. Whenever a discontinuity is encountered, fix it
        # it by adjusting the phase.
        if np.sum(np.abs(unitaries[i] - unitaries[i - 1])) > 1e-3:
            for j in range(3):
                if np.sum(np.abs(unitaries[i, j] - unitaries[i - 1, j])) > 1e-3:
                    phases = np.angle(unitaries[i, j] / unitaries[i - 1, j])
                    phase = next(
                        (ph for ph in phases if not np.isnan(ph) and abs(ph) > 1e-8), 0
                    )
                    unitaries[i, j] = np.exp(-1j * phase) * unitaries[i, j]
        mat = 1j * logm(unitaries[i] @ unitaries[i - 1].conj().T) / 1
        mat = 0.5 * (mat + mat.conj().T)
        hamiltonians[i - 1] = mat

    return hamiltonians, unitaries


def calculate_coeffs(operator):
    coeffs = [np.trace(x @ operator) / 2 for x in basis]
    with np.printoptions(precision=2):
        print(np.abs(coeffs))


def main():
    data = np.load("curve.npz")
    omega = data["omega"]
    tangent = data["tangent"][1:]

    # plt.plot(np.linspace(0, 1, omega.shape[0]), data["ratio"], label="ratio")
    # plt.grid()
    # plt.legend()
    # plt.show()

    #########################
    ## Extract the unitary and hamiltonian from the tangent.
    #########################
    # th = np.pi / 8
    # initial_u = (np.cos(th) * basis[0] + np.sin(th) * basis[2] + np.diag([0, 0, 1])) @ (
    #     basis[0] + np.diag([0, 0, 1])
    # )
    hamiltonians, unitaries = extract_pulse_v2(tangent)
    initial_u = unitaries[0]
    np.savez("unitaries.npz", uni=unitaries)

    # Verify that the hamiltonian simulation gives the same unitary
    hops = Ops(hamiltonians, np.arange(hamiltonians.shape[0]))
    uops = calculate_unitary(hops)

    with np.printoptions(precision=2, suppress=True):
        print("\nU from diagonalization:")
        print("t = 0:", unitaries[0], sep="\n")
        print("t = end:", unitaries[-10], sep="\n")
        print("U(t).U(0)dag", unitaries[-10] @ initial_u.conj().T, sep="\n")

        # Note that we do not need to compensate for U(0) in hamiltonian simulation.
        print("\nU from hamiltonian simulations:")
        print("t = 0:", normalize(uops.ops[0]), sep="\n")
        print("t = end:", np.abs(uops.ops[-10]), sep="\n")

    with np.printoptions(precision=5, suppress=True):
        print("\nHamiltonians")
        print("t = 0:", hops.ops[0], sep="\n")
        print("t = 500:", hops.ops[500], sep="\n")
        print("t = 1000:", hops.ops[-1], sep="\n")

    calc_tangent = np.empty_like(tangent)
    ham_coeffs = np.empty_like(tangent)
    diag_unitary_coeffs = np.empty_like(tangent)
    sim_unitary_coeffs = np.empty_like(tangent)
    for i in range(len(unitaries) - 1):
        operator = unitaries[i].conj().T @ basis[2] @ unitaries[i]
        calc_tangent[i] = np.abs(np.trace(basis @ operator, axis1=1, axis2=2)) / 2
        ham_coeffs[i] = np.abs(np.trace(basis @ hamiltonians[i], axis1=1, axis2=2)) / 2
        diag_unitary_coeffs[i] = (
            np.abs(
                np.trace(basis @ unitaries[i] @ initial_u.conj().T, axis1=1, axis2=2)
            )
            / 2
        )
        sim_unitary_coeffs[i] = (
            np.abs(np.trace(basis @ uops.ops[i], axis1=1, axis2=2)) / 2
        )

    ######################
    ##  Plot results
    ######################
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    # Plot tangent
    for i in range(calc_tangent.shape[1]):
        color = plt.cm.tab10(i % 10)
        axs[0, 0].plot(
            np.linspace(0, 1, tangent.shape[0]),
            np.abs(calc_tangent[:, i]),
            label=f"{i}",
            color=color,
            linestyle="-",
        )
        axs[0, 0].plot(
            np.linspace(0, 1, tangent.shape[0]),
            np.abs(tangent[:, i]),
            color=color,
            linestyle="--",
        )
    axs[0, 0].grid()
    axs[1, 0].set_ylim((0, np.max(tangent[:-10])))
    axs[0, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    axs[0, 0].set_title("Tangent Components")
    axs[0, 0].set_ylabel("Value")

    # Plot diagonalized unitaries
    for i in range(calc_tangent.shape[1]):
        color = plt.cm.tab10(i % 10)
        axs[0, 1].plot(
            np.linspace(0, 1, tangent.shape[0]),
            np.abs(diag_unitary_coeffs[:, i]),
            label=f"{i}",
            color=color,
            linestyle="-",
        )
    axs[0, 1].grid()
    axs[0, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    axs[0, 1].set_title("(Diagonalized) Unitary Coefficients")
    axs[0, 1].set_ylabel("Value")

    # Plot simulated unitaries
    for i in range(calc_tangent.shape[1]):
        color = plt.cm.tab10(i % 10)
        axs[1, 1].plot(
            np.linspace(0, 1, tangent.shape[0]),
            np.abs(sim_unitary_coeffs[:, i]),
            label=f"{i}",
            color=color,
            linestyle="-",
        )
    axs[1, 1].grid()
    axs[1, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    axs[1, 1].set_title("(Simulated) Unitary Coefficients")
    axs[1, 1].set_xlabel("Normalized Time")
    axs[1, 1].set_ylabel("Value")

    # Plot hamiltonian
    for i in range(calc_tangent.shape[1]):
        color = plt.cm.tab10(i % 10)
        axs[1, 0].plot(
            np.linspace(0, 1, tangent.shape[0]),
            np.abs(ham_coeffs[:, i]),
            label=f"{i}",
            color=color,
            linestyle="-",
        )
    axs[1, 0].grid()
    # axs[1, 0].set_ylim((0, np.max(ham_coeffs[:-10])))
    axs[1, 0].set_ylim((0, 0.02))
    axs[1, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    axs[1, 0].set_title("Hamiltonian Coefficients")
    axs[1, 0].set_xlabel("Normalized Time")
    axs[1, 0].set_ylabel("Value")

    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main()
