import hashlib
import itertools
import random
import unittest
import numpy as np
from sha3lab.keccak import sha3, permute, permute_batch, round_batch, messages128_states, weight
from sha3lab.hypercomplex import (encode,decode,round_spin,geometric_product,
    conjugate,squared_displacement,chi_spin)
from sha3lab.graphs import tarjan,solve_2sat,chi_relation_analysis,dependency_graph
from sha3lab.metamarkov import DyadicMetaMarkov,quantize
from sha3lab.attacks import collision_certificate,truncated_preimage


class KeccakTests(unittest.TestCase):
    def test_hashlib_byte_aligned_crosscheck(self):
        rng = random.Random(918273)
        lengths = [0,1,7,8,71,72,73,103,104,105,127,128,135,136,137,143,144,145,271,272,273,1024]
        for d in (224,256,384,512):
            for n in lengths:
                msg = rng.randbytes(n)
                with self.subTest(d=d,n=n):
                    self.assertEqual(sha3(msg,d),getattr(hashlib,f"sha3_{d}")(msg).digest())

    def test_known_answer(self):
        self.assertEqual(sha3(b"").hex(),"a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a")
        self.assertEqual(sha3(b"abc").hex(),"3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532")

    def test_batch_scalar_all_widths(self):
        rng = np.random.default_rng(410)
        for w in (1,2,4,8,16,32,64):
            a = rng.bit_generator.random_raw((8,25)) & np.uint64((1 << w)-1)
            rounds = 12+2*(w.bit_length()-1)
            for r,start in ((1,0),(rounds,0),(2,rounds-2)):
                expected = np.array([permute(row,r,w,start) for row in a],dtype=np.uint64)
                np.testing.assert_array_equal(permute_batch(a,r,w,start),expected)

    def test_padding_batch(self):
        rng = np.random.default_rng(42)
        messages = rng.integers(0,256,(10,128),dtype=np.uint8)
        outputs = permute_batch(messages128_states(messages))
        for m,o in zip(messages,outputs):
            self.assertEqual(o[:4].astype("<u8").tobytes(),hashlib.sha3_256(m.tobytes()).digest())


class HypercomplexTests(unittest.TestCase):
    def test_generators(self):
        basis = np.eye(64,dtype=np.int64)
        for i in range(6):
            np.testing.assert_array_equal(geometric_product(basis[1 << i],basis[1 << i]),-basis[0])
            for j in range(i):
                np.testing.assert_array_equal(geometric_product(basis[1 << i],basis[1 << j]),
                    -geometric_product(basis[1 << j],basis[1 << i]))

    def test_associativity_and_conjugation(self):
        rng = np.random.default_rng(4)
        a,b,c = rng.integers(-1,2,(3,64))
        np.testing.assert_array_equal(geometric_product(geometric_product(a,b),c),
                                      geometric_product(a,geometric_product(b,c)))
        np.testing.assert_array_equal(conjugate(geometric_product(a,b)),geometric_product(conjugate(b),conjugate(a)))
        self.assertEqual(geometric_product(a,conjugate(b))[0],a@b)

    def test_exact_spin_round_and_metric(self):
        rng = np.random.default_rng(811)
        a,b = rng.bit_generator.random_raw((2,16,25))
        np.testing.assert_array_equal(decode(encode(a)),a)
        for ir in (0,1,7,23):
            np.testing.assert_array_equal(decode(round_spin(encode(a),ir)),round_batch(a,ir))
        np.testing.assert_array_equal(squared_displacement(a,b),4*weight(a ^ b))

    def test_chi_spin_truth_table(self):
        for a,b,c in itertools.product((0,1),repeat=3):
            out = a ^ ((1-b) & c)
            self.assertEqual(chi_spin(1-2*a,1-2*b,1-2*c),1-2*out)


class GraphTests(unittest.TestCase):
    def test_scc_against_reachability(self):
        rng = random.Random(91)
        for _ in range(50):
            g = [[j for j in range(9) if rng.random()<.2] for _ in range(9)]
            reach = []
            for source in range(9):
                seen,stack = {source},[source]
                while stack:
                    for j in g[stack.pop()]:
                        if j not in seen:
                            seen.add(j); stack.append(j)
                reach.append(seen)
            cid = {v:k for k,c in enumerate(tarjan(g)) for v in c}
            for a,b in itertools.product(range(9),repeat=2):
                self.assertEqual(cid[a]==cid[b],b in reach[a] and a in reach[b])

    def test_2sat_against_bruteforce(self):
        rng = random.Random(776)
        for _ in range(200):
            n = 6
            clauses = [tuple(rng.choice((-1,1))*rng.randint(1,n) for _ in range(rng.randint(1,2)))
                       for _ in range(rng.randint(1,24))]
            brute = [a for a in itertools.product((False,True),repeat=n)
                     if all(any(a[abs(lit)-1] == (lit>0) for lit in c) for c in clauses)]
            result = solve_2sat(n,clauses)
            self.assertEqual(result is None,not brute)
            if result is not None:
                self.assertIn(tuple(result),brute)
        with self.assertRaises(ValueError):
            solve_2sat(3,[(1,2,3)])

    def test_chi_majority_obstruction(self):
        result = chi_relation_analysis()
        self.assertEqual(result["nontrivial_implied_2clauses"],[])
        self.assertEqual(len(result["relaxation_assignments"]),16)
        self.assertNotIn(result["majority_counterexample"]["majority"],result["valid_assignments"])
        # b=1 simplifies chi to z=a: this genuine 2-SAT slice is solved exactly.
        solution = solve_2sat(2,[(-1,2),(1,-2),(1,)])
        self.assertEqual(solution,[True,True])

    def test_dependency_graph_support(self):
        rng = np.random.default_rng(134)
        a = rng.integers(0,2,(32,25),dtype=np.uint64)
        base = permute_batch(a,1,w=1)
        g = dependency_graph(1)
        for i in range(25):
            b = a.copy(); b[:,i] ^= np.uint64(1)
            observed = np.flatnonzero((base != permute_batch(b,1,w=1)).any(axis=0))
            self.assertTrue(set(observed).issubset(g[i]))


class ExperimentTests(unittest.TestCase):
    def test_universal_one_round_collision(self):
        cert = collision_certificate()
        self.assertEqual(cert["reduced_digest_a"],cert["reduced_digest_b"])
        self.assertNotEqual(cert["full_sha3_256_a"],cert["full_sha3_256_b"])
        rng = np.random.default_rng(999)
        a = messages128_states(rng.integers(0,256,(128,128),dtype=np.uint8))
        b = a.copy(); b[:,5] ^= np.uint64(1); b[:,10] ^= np.uint64(1)
        np.testing.assert_array_equal(permute_batch(a,1)[:,:4],permute_batch(b,1)[:,:4])

    def test_meta_probabilities_and_frozen_counts(self):
        seq = quantize(np.random.default_rng(3).integers(0,1601,(40,8)))
        model = DyadicMetaMarkov(cap=3).fit(seq[:20])
        before = {k:v.copy() for k,v in model.counts.items()}
        losses = model.losses(seq[20:])
        self.assertTrue(np.isfinite(losses).all())
        self.assertAlmostEqual(model.predict(0,3,1).sum(),1)
        self.assertTrue((model.predict(0,3,1)>0).all())
        for k,v in before.items():
            np.testing.assert_array_equal(v,model.counts[k])
            self.assertLessEqual(int(v.max()),3)

    def test_search_certificate_and_reproducibility(self):
        for method in ("random","meta"):
            a = truncated_preimage(5,24,818,64,4,method)
            b = truncated_preimage(5,24,818,64,4,method)
            self.assertEqual(a,b)
            val = int.from_bytes(hashlib.sha3_256(bytes.fromhex(a["message_hex"])).digest()[:2],"little") & 15
            self.assertEqual(a["value"],val)
            self.assertEqual(a["best_hamming"],(val ^ 5).bit_count())
            self.assertLessEqual(a["queries"],64)


if __name__ == "__main__":
    unittest.main()
