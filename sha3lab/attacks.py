"""Constructive reduced-round collision and budgeted truncated preimages."""
from __future__ import annotations
import hashlib
import math
import numpy as np
from .keccak import sha3
from .graphs import tarjan


def collision_certificate():
    a = bytes(128)
    b = bytearray(a)
    b[40] ^= 1                     # (x,y,z) = (0,1,0)
    b[80] ^= 1                     # (x,y,z) = (0,2,0)
    b = bytes(b)
    h1, h2 = sha3(a, rounds=1), sha3(b, rounds=1)
    assert h1 == h2 and a != b
    return {"definition": "SHA3-256 padding/rate, PREFIX round RC[0] only",
            "message_a_hex": a.hex(), "message_b_hex": b.hex(),
            "reduced_digest_a": h1.hex(), "reduced_digest_b": h2.hex(),
            "full_sha3_256_a": hashlib.sha3_256(a).hexdigest(),
            "full_sha3_256_b": hashlib.sha3_256(b).hexdigest()}


class FeedbackPolicy:
    """Finite dyadic contexts, capped counts, and self-prediction feedback."""
    def __init__(self, target_bits, feedback=True):
        self.bits = target_bits
        self.use_feedback = feedback
        self.prior = np.array([math.comb(target_bits,k)/2**target_bits
                               for k in range(target_bits+1)])
        self.counts = {}
        self.feedback = 0

    def keys(self, h, action):
        return [(shift, h >> shift, self.feedback, action) for shift in (4,3,2,1,0)]

    def predict(self, h, action):
        p = self.prior.copy()
        for key in self.keys(h,action):
            c = self.counts.get(key)
            if c is not None:
                p = (c+16*p)/(c.sum()+16)
        return p

    def update(self, h, action, new_h, prediction):
        for key in self.keys(h,action):
            c = self.counts.setdefault(key,np.zeros(self.bits+1,dtype=np.uint16))
            c[new_h] = min(255,int(c[new_h])+1)
        self.feedback = int(np.argmax(prediction) == new_h) if self.use_feedback else 0


def truncated_preimage(target, rounds, seed, budget=2048, target_bits=12, method="random"):
    """Every distinct queried message counts, including restarts.

    Hypercomplex squared displacement equals 4*Hamming distance exactly.
    Consequently the integer Hamming implementation is the exact fast path
    for that score, not an approximation. Twelve-bit output truncation is a
    benchmark, not a preimage of the 256-bit digest.
    """
    if method not in ("random","meta","hillclimb","meta_no_feedback","meta_no_scc") or not 1 <= target_bits <= 16 or budget < 1:
        raise ValueError("invalid benchmark configuration")
    rng = np.random.default_rng(seed)
    mask = (1 << target_bits)-1
    def digest(m):
        raw = hashlib.sha3_256(m).digest() if rounds == 24 else sha3(m, rounds=rounds)
        return int.from_bytes(raw[:2],"little") & mask
    def fresh():
        return rng.integers(0,256,128,dtype=np.uint8).tobytes()
    policy = FeedbackPolicy(target_bits, feedback=method != "meta_no_feedback")
    seen = set()
    graph = [set() for _ in range(target_bits+1)]
    current, h = fresh(), target_bits
    best, best_message = target_bits+1, None
    forced_restart, stalls, restarts, scc_checks = True, 0, 0, 0
    for query in range(1,budget+1):
        action, prediction = 4, None
        if method == "random" or forced_restart:
            candidate = fresh()
            forced_restart = False
        else:
            predictions = [policy.predict(h,a) for a in range(5)]
            utilities = np.array([p[:h].sum() for p in predictions])
            if method == "hillclimb":
                action = int(rng.integers(5))
            elif rng.random() < 0.25:
                action = int(rng.integers(5))
            else:
                action = int(np.argmax(utilities + rng.random(5)*1e-10))
            prediction = predictions[action]
            if action == 4:
                candidate = fresh()
            else:
                buf = bytearray(current)
                for bit in rng.choice(1024,size=1 << action,replace=False):
                    buf[int(bit)//8] ^= 1 << (int(bit) % 8)
                candidate = bytes(buf)
        if candidate in seen:
            action, prediction = 4, None
            while candidate in seen:
                candidate = fresh()
        seen.add(candidate)
        value = digest(candidate)
        new_h = (value ^ target).bit_count()
        if new_h < best:
            best, best_message, stalls = new_h, candidate, 0
        else:
            stalls += 1
        if new_h == 0:
            return {"success": True,"queries": query,"best_hamming": 0,
                    "message_hex": candidate.hex(),"value": value,
                    "scc_checks": scc_checks,"forced_restarts": restarts}
        if method != "random":
            if prediction is None:
                prediction = policy.predict(h,action)
            if method != "hillclimb":
                policy.update(h,action,new_h,prediction)
            accept = action == 4 or new_h <= h or rng.random() < 0.03
            graph[h].add(new_h if accept else h)
            if accept:
                current, h = candidate,new_h
            if stalls >= 32 and query % 16 == 0:
                recurrent = True
                if method not in ("hillclimb","meta_no_scc"):
                    scc_checks += 1
                    comp = next(c for c in tarjan([sorted(g) for g in graph]) if h in c)
                    recurrent = len(comp) > 1 or h in graph[h]
                if recurrent:
                    forced_restart, stalls = True,0
                    restarts += 1
    return {"success": False,"queries": budget,"best_hamming": best,
            "message_hex": best_message.hex(),
            "value": digest(best_message),"scc_checks": scc_checks,
            "forced_restarts": restarts}
