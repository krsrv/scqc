import numpy as np
import scipy


class Ops:
    def __init__(self, ops: np.ndarray, t_range: np.ndarray):
        self.ops = ops
        self.t_range = t_range

    def __add__(self, other):
        if not isinstance(other, Ops):
            return NotImplemented
        # Elementwise sum of ops, t_range should match
        if not np.array_equal(self.t_range, other.t_range):
            raise ValueError("t_range arrays do not match")
        return Ops(self.ops + other.ops, self.t_range)


def calculate_unitary(hamiltonians: Ops) -> Ops:
    out = np.empty(
        (len(hamiltonians.t_range) + 1, *hamiltonians.ops[0].shape), dtype=np.complex128
    )
    out[0] = np.eye(hamiltonians.ops[0].shape[0], dtype=np.complex128)
    dt = hamiltonians.t_range[1] - hamiltonians.t_range[0]
    for i in range(len(hamiltonians.ops)):
        out[i + 1] = scipy.linalg.expm(-1j * hamiltonians.ops[i] * dt) @ out[i]
    return Ops(out, np.append(hamiltonians.t_range, [hamiltonians.t_range[-1] + dt]))


def first_order_error(ut: Ops, error: Ops):
    dt = ut.t_range[1] - ut.t_range[0]
    e = error.ops[0]
    # Calculate first order error integral
    out = np.zeros_like(error.ops[0])
    for i in range(len(ut.ops)):
        out += (ut.ops[i].conj().T @ e @ ut.ops[i]) * dt
    return out


def second_order_error(ut: Ops, error: Ops):
    dt = ut.t_range[1] - ut.t_range[0]
    h_int = np.zeros_like(ut.ops)
    e = error.ops[0]
    for i in range(len(h_int)):
        h_int[i] = ut.ops[i].conj().T @ e @ ut.ops[i]
    h_int_cumsum = np.cumsum(h_int, axis=0) * dt
    # Calculate second order error integral
    out = np.zeros_like(error.ops[0])
    for i in range(len(ut.ops)):
        out += (h_int[i] @ h_int_cumsum[i] - h_int_cumsum[i] @ h_int[i]) * dt
    return out / 2


def normalize(A: np.ndarray):
    threshold = 1e-3
    A1 = np.where(np.abs(A) < threshold, 0, A).reshape(-1)
    if np.all(A1 == 0):
        # raise ValueError("A1 is zero; cannot normalize.")
        return A
    i = np.flatnonzero(A1)[0]
    phase = A1[i] / np.abs(A1[i])
    return A / phase


def fidelity(U, U_ideal):
    return np.abs(np.trace(U.conj().T @ U_ideal))


def rotated_fidelity(U, U_ideal):
    return np.trace(np.abs(U.conj().T @ U_ideal))
