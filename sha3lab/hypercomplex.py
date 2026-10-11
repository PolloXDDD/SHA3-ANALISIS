"""An exact Cl(0,6) representation of each 64-bit Keccak lane.

This is a real associative Clifford algebra, with e_i**2 = -1 and
e_i e_j = -e_j e_i for i != j. Boolean operations act COEFFICIENTWISE.
The geometric product is separately implemented for genuine multivector
features; it is not substituted for AND or XOR.
"""
from __future__ import annotations
import numpy as np
from .keccak import RC, RHO, bits


def blade_sign(a: int, b: int) -> int:
    inversions = sum((b & ((1 << j)-1)).bit_count()
                     for j in range(6) if (a >> j) & 1)
    return -1 if (inversions + (a & b).bit_count()) % 2 else 1


XOR_INDEX = np.bitwise_xor.outer(np.arange(64), np.arange(64))
PRODUCT_SIGNS = np.array([[blade_sign(i, i ^ k) for i in range(64)]
                         for k in range(64)], dtype=np.int8)
GRADES = np.array([i.bit_count() for i in range(64)])
CONJ_SIGNS = np.array([(-1) ** (k*(k+1)//2) for k in GRADES], dtype=np.int8)


def geometric_product(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, dtype=np.int64),
                               np.asarray(b, dtype=np.int64))
    if a.shape[-1] != 64:
        raise ValueError("Cl(0,6) needs 64 coefficients")
    return np.sum(a[..., None, :] * b[..., XOR_INDEX] * PRODUCT_SIGNS, axis=-1)


def conjugate(a):
    return np.asarray(a) * CONJ_SIGNS


def encode(state):
    return 1 - 2 * bits(state).astype(np.int64)


def decode(q):
    q = np.asarray(q)
    if q.shape[-2:] != (25, 64) or not np.isin(q, (-1, 1)).all():
        raise ValueError("expected 25 lanes of Boolean spin coefficients")
    bb = ((1-q)//2).astype(np.uint64)
    return np.sum(bb << np.arange(64, dtype=np.uint64), axis=-1, dtype=np.uint64)


def chi_spin(a, b, c):
    return a * (1-b+c+b*c) // 2


def round_spin(q, ir):
    """Exact transported round using integer spin-polynomial operations."""
    q = np.asarray(q, dtype=np.int64)
    c = np.stack([np.prod(q[..., x::5, :], axis=-2) for x in range(5)], axis=-2)
    d = [c[..., (x-1) % 5, :] * np.roll(c[..., (x+1) % 5, :], 1, axis=-1)
         for x in range(5)]
    b = np.empty_like(q)
    for y in range(5):
        for x in range(5):
            j = x + 5*y
            b[..., y+5*((2*x+3*y) % 5), :] = np.roll(q[..., j, :]*d[x], RHO[j], axis=-1)
    out = np.empty_like(q)
    for y in range(5):
        for x in range(5):
            out[..., x+5*y, :] = chi_spin(b[..., x+5*y, :],
                b[..., (x+1) % 5+5*y, :], b[..., (x+2) % 5+5*y, :])
    rc = np.array([1-2*((RC[ir] >> z) & 1) for z in range(64)])
    out[..., 0, :] *= rc
    return out


def squared_displacement(a, b):
    return np.sum((encode(a)-encode(b))**2, axis=(-1, -2))


def grade_features(delta):
    """Seven grade energies from four fixed adjacent/column lane pairs.

    Pair choice is fixed before experiments. Log scaling controls magnitude.
    Features are basis-dependent; no invariant status is asserted.
    """
    q = encode(delta)
    energy = np.zeros(q.shape[:-2] + (7,), dtype=float)
    for i, j in ((0, 1), (5, 10), (10, 15), (15, 20)):
        p = geometric_product(q[..., i, :], conjugate(q[..., j, :]))
        for g in range(7):
            energy[..., g] += np.sum(p[..., GRADES == g]**2, axis=-1)
    return np.log1p(energy)
