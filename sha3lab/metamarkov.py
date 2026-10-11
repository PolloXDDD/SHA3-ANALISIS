"""Finite self-referential dyadic Markov predictor for diffusion trajectories.

The dyadic hierarchy is an explicit interpretation of a finite fractal, not
an assertion of fractal dynamics in Keccak. Counts are capped; the feedback
bit is produced by the model's own previous prediction. Parameters freeze
before held-out evaluation. Round phase is included in every context.
"""
from __future__ import annotations
import numpy as np


def quantize(weights, width=1600, bins=32):
    return np.minimum(bins-1, np.asarray(weights, dtype=int)*bins//(width+1))


class DyadicMetaMarkov:
    def __init__(self, bins=32, feedback=True, tau=32.0, cap=65535):
        if bins < 2 or bins & (bins-1):
            raise ValueError("bins must be a power of two")
        self.bins, self.feedback, self.tau, self.cap = bins, feedback, tau, cap
        self.levels = bins.bit_length()-1
        self.counts = {}
        self.prior = None

    def _keys(self, phase, state, feedback):
        e = int(feedback) if self.feedback else 0
        return [(phase, l, int(state) >> (self.levels-l), e)
                for l in range(self.levels+1)]

    def predict(self, phase, state, feedback=0):
        if self.prior is None:
            raise RuntimeError("fit first")
        p = self.prior[phase].copy()
        for key in self._keys(phase,state,feedback):
            c = self.counts.get(key)
            if c is not None:
                p = (c+self.tau*p)/(c.sum()+self.tau)
        return p

    def fit(self, sequences):
        seq = np.asarray(sequences,dtype=int)
        self.counts = {}
        self.prior = np.full((seq.shape[1]-1,self.bins),0.5)
        for r in range(seq.shape[1]-1):
            self.prior[r] += np.bincount(seq[:,r+1],minlength=self.bins)
        self.prior /= self.prior.sum(axis=1,keepdims=True)
        for row in seq:
            e = 0
            for r,(x,y) in enumerate(zip(row[:-1],row[1:])):
                p = self.predict(r,x,e)
                for key in self._keys(r,x,e):
                    c = self.counts.setdefault(key,np.zeros(self.bins,dtype=np.uint32))
                    c[y] = min(int(c[y])+1,self.cap)
                e = int(np.argmax(p) == y)
        return self

    def losses(self, sequences):
        losses = []
        for row in np.asarray(sequences,dtype=int):
            e, loss = 0, 0.0
            for r,(x,y) in enumerate(zip(row[:-1],row[1:])):
                p = self.predict(r,x,e)
                loss -= np.log2(p[y])
                e = int(np.argmax(p) == y)
            losses.append(loss/(len(row)-1))
        return np.array(losses)


def control_losses(train,test,bins=32):
    phase = np.full((train.shape[1]-1,bins),0.5)
    first = np.full((train.shape[1]-1,bins,bins),0.5)
    for row in train:
        for r,(x,y) in enumerate(zip(row[:-1],row[1:])):
            phase[r,y] += 1
            first[r,x,y] += 1
    phase /= phase.sum(axis=1,keepdims=True)
    first /= first.sum(axis=2,keepdims=True)
    pp,ff = [],[]
    for row in test:
        pp.append(np.mean([-np.log2(phase[r,y]) for r,y in enumerate(row[1:])]))
        ff.append(np.mean([-np.log2(first[r,x,y]) for r,(x,y) in enumerate(zip(row[:-1],row[1:]))]))
    return np.array(pp),np.array(ff)
