"""Tarjan SCC, an exact 2-SAT solver, and coordinate dependency graphs."""
from __future__ import annotations
from itertools import product, combinations
from .keccak import RHO


def tarjan(graph):
    """Iterative Tarjan, O(V+E); vertices are 0,...,len(graph)-1."""
    n = len(graph)
    index = [-1] * n
    low = [0] * n
    active = [False] * n
    stack, components = [], []
    counter = 0
    for root in range(n):
        if index[root] != -1:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        active[root] = True
        dfs = [(root, iter(graph[root]))]
        while dfs:
            v, children = dfs[-1]
            try:
                u = next(children)
            except StopIteration:
                dfs.pop()
                if low[v] == index[v]:
                    comp = []
                    while True:
                        u = stack.pop()
                        active[u] = False
                        comp.append(u)
                        if u == v:
                            break
                    components.append(comp)
                if dfs:
                    p = dfs[-1][0]
                    low[p] = min(low[p], low[v])
                continue
            if index[u] == -1:
                index[u] = low[u] = counter
                counter += 1
                stack.append(u)
                active[u] = True
                dfs.append((u, iter(graph[u])))
            elif active[u]:
                low[v] = min(low[v], index[u])
    return components


def solve_2sat(n, clauses):
    """Signed 1-based literals; clauses length 1 or 2. None means UNSAT."""
    graph = [[] for _ in range(2*n)]
    def node(lit):
        if not 1 <= abs(lit) <= n:
            raise ValueError("invalid literal")
        return 2*(abs(lit)-1) + int(lit < 0)
    for clause in clauses:
        if not clause:
            return None
        if len(clause) > 2:
            raise ValueError("a nonlinear clause cannot be silently treated as 2-SAT")
        a, b = node(clause[0]), node(clause[-1])
        graph[a ^ 1].append(b)
        graph[b ^ 1].append(a)
    cid = [0] * (2*n)
    for i, comp in enumerate(tarjan(graph)):
        for v in comp:
            cid[v] = i
    if any(cid[2*v] == cid[2*v+1] for v in range(n)):
        return None
    assignment = [cid[2*v] < cid[2*v+1] for v in range(n)]
    assert all(any(assignment[abs(lit)-1] == (lit > 0) for lit in c) for c in clauses)
    return assignment


def dependency_graph(w=64):
    """One-round structural support, with input/output coordinates identified.

    Theta supports are exact (symmetric differences); chi takes unions.
    The identified graph can have cycles. The time-unrolled circuit is a DAG.
    """
    before = [[set() for _ in range(w)] for _ in range(25)]
    for y in range(5):
        for x in range(5):
            j = x+5*y
            for z in range(w):
                support = {j*w+z}
                for yy in range(5):
                    support.symmetric_difference_update({((x-1) % 5+5*yy)*w+z})
                    support.symmetric_difference_update({((x+1) % 5+5*yy)*w+(z-1) % w})
                dest = y+5*((2*x+3*y) % 5)
                before[dest][(z+RHO[j]) % w] = support
    graph = [set() for _ in range(25*w)]
    for y in range(5):
        for x in range(5):
            for z in range(w):
                supp = set().union(*(before[(x+d) % 5+5*y][z] for d in range(3)))
                for source in supp:
                    graph[source].add((x+5*y)*w+z)
    return [sorted(g) for g in graph]


def chi_relation_analysis():
    """Exhaustive local chi relation z = a XOR ((NOT b) AND c)."""
    relation = [(a,b,c,a ^ ((1-b) & c)) for a,b,c in product((0,1), repeat=3)]
    valid = set(relation)
    implied = []
    literals = list(range(1,5)) + list(range(-4,0))
    for k in (1,2):
        for clause in combinations(literals,k):
            if k == 2 and clause[0] == -clause[1]:
                continue
            if all(any(row[abs(lit)-1] == int(lit>0) for lit in clause) for row in relation):
                implied.append(clause)
    witness = None
    for triple in combinations(relation,3):
        maj = tuple(int(sum(t[j] for t in triple) >= 2) for j in range(4))
        if maj not in valid:
            witness = {"satisfying_triples": triple, "majority": maj}
            break
    return {"valid_assignments": relation, "nontrivial_implied_2clauses": implied,
            "majority_counterexample": witness,
            "relaxation_assignments": [v for v in product((0,1), repeat=4)
              if all(any(v[abs(lit)-1] == int(lit>0) for lit in c) for c in implied)]}
