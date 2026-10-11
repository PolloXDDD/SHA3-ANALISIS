"""Readable scalar and batched Keccak implementations.

Lanes are indexed x + 5*y; bits inside a lane are little endian.
Reduced experiments use PREFIX rounds 0,...,r-1, NOT the last-r-round
Keccak-p convention. Pass start=24-r (w=64) explicitly for that convention.
Only byte-aligned SHA-3 messages are supported.
"""
from __future__ import annotations

import numpy as np

RC = (
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
    0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
)
RHO = (0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43,
       25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14)


def check_schedule(w: int, rounds: int, start: int) -> None:
    if w not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError("w must be a power of two between 1 and 64")
    maximum = 12 + 2 * (w.bit_length() - 1)
    if not (0 <= start <= start + rounds <= maximum):
        raise ValueError("round schedule exceeds Keccak-f width")


def rot(a: int, n: int, w: int) -> int:
    n %= w
    return ((a << n) | (a >> ((w - n) % w))) & ((1 << w) - 1)


def round_scalar(state, ir: int, w: int = 64) -> list[int]:
    mask = (1 << w) - 1
    a = [int(v) & mask for v in state]
    c = [a[x] ^ a[x+5] ^ a[x+10] ^ a[x+15] ^ a[x+20] for x in range(5)]
    d = [c[(x-1) % 5] ^ rot(c[(x+1) % 5], 1, w) for x in range(5)]
    b = [0] * 25
    for y in range(5):
        for x in range(5):
            j = x + 5*y
            b[y + 5*((2*x+3*y) % 5)] = rot(a[j] ^ d[x], RHO[j], w)
    out = [b[x+5*y] ^ ((~b[(x+1) % 5+5*y]) & b[(x+2) % 5+5*y])
           for y in range(5) for x in range(5)]
    out[0] ^= RC[ir] & mask
    return out


def permute(state, rounds: int = 24, w: int = 64, start: int = 0):
    check_schedule(w, rounds, start)
    if len(state) != 25:
        raise ValueError("expected 25 lanes")
    out = list(state)
    for ir in range(start, start + rounds):
        out = round_scalar(out, ir, w)
    return out


def round_batch(state: np.ndarray, ir: int, w: int = 64) -> np.ndarray:
    """State shape (...,25), uint64; returns a new array."""
    a = np.asarray(state, dtype=np.uint64)
    if a.shape[-1] != 25:
        raise ValueError("expected 25 lanes")
    mask = np.uint64((1 << w) - 1)
    a = a & mask
    c = np.stack([np.bitwise_xor.reduce(a[..., x::5], axis=-1)
                  for x in range(5)], axis=-1)
    d = []
    for x in range(5):
        z = c[..., (x+1) % 5]
        rz = ((z << np.uint64(1 % w)) | (z >> np.uint64((w-1) % w))) & mask
        d.append(c[..., (x-1) % 5] ^ rz)
    b = np.empty_like(a)
    for y in range(5):
        for x in range(5):
            j = x + 5*y
            z = a[..., j] ^ d[x]
            s = RHO[j] % w
            b[..., y+5*((2*x+3*y) % 5)] = (
                (z << np.uint64(s)) | (z >> np.uint64((w-s) % w))) & mask
    out = np.empty_like(a)
    for y in range(5):
        for x in range(5):
            out[..., x+5*y] = b[..., x+5*y] ^ (
                (~b[..., (x+1) % 5+5*y]) & b[..., (x+2) % 5+5*y])
    out[..., 0] ^= np.uint64(RC[ir] & int(mask))
    return out


def permute_batch(state, rounds=24, w=64, start=0):
    check_schedule(w, rounds, start)
    out = np.asarray(state, dtype=np.uint64).copy()
    for ir in range(start, start + rounds):
        out = round_batch(out, ir, w)
    return out


def sha3(message: bytes, digest_bits: int = 256, rounds: int = 24) -> bytes:
    if digest_bits not in (224, 256, 384, 512):
        raise ValueError("invalid SHA-3 output length")
    check_schedule(64, rounds, 0)
    rate = (1600 - 2*digest_bits) // 8
    data = bytearray(message)
    data.append(0x06)
    data.extend(b"\x00" * ((-len(data)) % rate))
    data[-1] |= 0x80
    a = [0] * 25
    for off in range(0, len(data), rate):
        for j in range(rate // 8):
            a[j] ^= int.from_bytes(data[off+8*j:off+8*j+8], "little")
        a = permute(a, rounds)
    return b"".join(v.to_bytes(8, "little") for v in a)[:digest_bits // 8]


def bits(state: np.ndarray, w: int = 64) -> np.ndarray:
    a = np.asarray(state, dtype=np.uint64)
    return ((a[..., :, None] >> np.arange(w, dtype=np.uint64)) & 1).astype(np.uint8)


def weight(state: np.ndarray) -> np.ndarray:
    return np.bitwise_count(np.asarray(state, dtype=np.uint64)).sum(axis=-1)


def messages128_states(messages: np.ndarray) -> np.ndarray:
    """Padded pre-permutation states for 128-byte SHA3-256 inputs."""
    m = np.ascontiguousarray(messages, dtype=np.uint8)
    if m.ndim != 2 or m.shape[1] != 128:
        raise ValueError("expected (n,128) input bytes")
    a = np.zeros((len(m), 25), dtype=np.uint64)
    a[:, :16] = m.view("<u8").reshape(-1, 16)
    a[:, 16] = np.uint64(0x8000000000000006)
    return a
