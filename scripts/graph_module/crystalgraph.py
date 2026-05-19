import itertools
import functools
import networkx as nx
import numpy as np
import sympy as sp
from collections import deque


def remove_selfloops(G):
    newG = G.copy()
    loops = list(nx.selfloop_edges(G))
    newG.remove_edges_from(loops)
    return newG

def graph_embedding(graph, wrap=True):
    """
    standard embedding of quotient graph.
    return scaled positions and modified graph
    """
    assert nx.number_connected_components(graph) == 1, "The input graph should be connected!"

    if len(graph) == 1:
        return np.array([[0,0,0]]), graph

    ## Laplacian Matrix
    lapMat = nx.laplacian_matrix(graph).todense()
    invLap = np.linalg.inv(lapMat[1:,1:])
    sMat = np.zeros((len(graph), 3))
    for edge in graph.edges(data=True):
        _,_,data = edge
        i,j = data['direction']
        sMat[i] += data['vector']
        sMat[j] -= data['vector']
    pos = np.dot(invLap,sMat[1:])
    pos = np.concatenate(([[0,0,0]], pos))
    pos = np.array(pos)
    # remove little difference
    pos[np.abs(pos)<1e-4] = 0
    newG = graph.copy()
    if wrap:
    # solve offsets
        offsets = np.floor(pos).astype(np.int32)
        pos -= offsets
        for edge in newG.edges(data=True, keys=True):
            m,n, key, data = edge
            i,j = data['direction']
            ## modify the quotient graph according to offsets
            newG[m][n][key]['vector'] = offsets[j] - offsets[i] + data['vector']

    return pos, newG

def sym_graph_embedding(graph):
    assert nx.number_connected_components(graph) == 1, "The input graph should be connected!"
   
    N = len(graph) # Numeber of atoms
    if N == 1:
        return [0,0,0]

    ## Laplacian Matrix
    lapMat = nx.laplacian_matrix(graph).todense()
    sMat = np.zeros((N, 3), dtype=np.int8)
    for edge in graph.edges(data=True):
        _,_,data = edge
        i,j = data['direction']
        sMat[i] += data['vector']
        sMat[j] -= data['vector']
    
    xs = sp.symbols("x1:{}".format(N))
    coefs = lapMat[1:,1:]
    posArr = sp.zeros(N,3)
    for ax in range(3):
        matrix = sp.Matrix(np.concatenate((coefs, sMat[1:,ax:ax+1]),axis=1))
        res = sp.solve_linear_system_LU(matrix, xs)
        for i in range(N-1):
            posArr[i+1,ax] = res[xs[i]]
    
    return posArr

def barycenter_disp(atoms, graph, selfloop=True):
    """
    Compute the displacement of every atom from the barycenter of neighboring atoms.
    atoms: ASE Atoms object, the structure
    graph: the quotient graph related to atoms
    selfloop: consider self loops in the quotient graph or not
    """

    assert len(atoms) == graph.number_of_nodes(), "The number of atoms should equal to number of nodes in graph!"

    pos = atoms.get_scaled_positions()
    # Sum of neighboring positions
    sumPos = np.zeros_like(pos)
    degrees = np.zeros(len(atoms))
    if selfloop:
        G = graph
    else:
        G = remove_selfloops(graph)
    for edge in G.edges(data=True):
        _,_,data = edge
        i,j = data['direction']
        sumPos[i] += pos[j] + data['vector']
        sumPos[j] += pos[i] - data['vector']
        degrees[i] += 1
        degrees[j] += 1
    
    barycenters = sumPos / np.expand_dims(degrees,1)
    disp = pos - barycenters

    return disp

def _generic_bfs_edges(G, source, neighbors=None, depth_limit=None, sort_neighbors=None):
    """
    from NetworkX's generic_bfs_edges, but slightly change sort_neighbors
    """
    if callable(sort_neighbors):
        _neighbors = neighbors
        neighbors = lambda node: iter(sort_neighbors(_neighbors(node), node))

    if neighbors is None:
        neighbors = G.neighbors

    visited = {source}
    if depth_limit is None:
        depth_limit = len(G)
    queue = deque([(source, depth_limit, neighbors(source))])
    while queue:
        parent, depth_now, children = queue[0]
        try:
            child = next(children)
            if child not in visited:
                yield parent, child
                visited.add(child)
                if depth_now > 1:
                    queue.append((child, depth_now - 1, neighbors(child)))
        except StopIteration:
            queue.popleft()

def _bfs_edges(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    """
    from NetworkX's bfs_edges, but use _generic_bfs_edges instead
    """
    if reverse and G.is_directed():
        successors = G.predecessors
    else:
        successors = G.neighbors
    yield from _generic_bfs_edges(G, source, successors, depth_limit, sort_neighbors)

def bfs_sorting(atoms, graph, baryPos=False):
    """
    Get all the permutations based on canonical forms and BFS.
    Return `len(atoms)` permutations.
    """
    assert len(atoms) == graph.number_of_nodes(), "The number of atoms should equal to number of nodes in graph!"
    assert nx.number_connected_components(graph) == 1, "The graph should be connected!"


    N = len(atoms)
    if baryPos is False:
        baryPos = sym_graph_embedding(graph)
    # realPos = atoms.get_scaled_positions()
    # realPos -= realPos[0] # make translation so that the first atomic position is (0,0,0)
    # diffPos = realPos - np.array(baryPos).astype(realPos.dtype)
    diffPos = barycenter_disp(atoms, graph, selfloop=False)

    # graph preprocessing
    G = nx.Graph(remove_selfloops(graph))

    # sort func
    def cmp_vec(a, b):
        """
        Compare two vectors by lexicographical order
        """
        
        if a[0] < b[0]:
            return -1
        elif a[0] > b[0]:
            return 1
        elif a[1] < b[1]:
            return -1
        elif a[1] > b[1]:
            return 1
        elif a[2] < b[2]:
            return -1
        elif a[2] > b[2]:
            return 1
        else :
            return 0

        
    def sort_func(neighbors, s):
        nei = list(neighbors)
        allD = {}
        for i in nei:
            # Compare displacements of parallel edges, choose the one with minimal elements
            minD = sp.Matrix([2,2,2])
            edges = graph[s][i]
            for _, val in edges.items():
                vec = val['vector']
                if val['direction'] == (s,i):
                    D = baryPos.row(i) - baryPos.row(s) + sp.Matrix([[int(v) for v in vec]])
                else:
                    D = baryPos.row(i) - baryPos.row(s) - sp.Matrix([[int(v) for v in vec]])
                D_remainder = D.applyfunc(lambda x: x % 1)
                if cmp_vec(D_remainder,minD) == -1:
                    minD = D_remainder
            allD[i] = minD
        
        def cmp_neighbors(m,n):
            """
            Compare two atoms.
            Firstly compare their barycentric position
            If they are same, then compare diffPos
            """
            baryCmp = cmp_vec(allD[m], allD[n])
            if baryCmp == 0:
                diff = diffPos[m] - diffPos[n]
                if diff[~np.isclose(diffPos[m],diffPos[n])][0] < 0:
                    return -1
                else:
                    return 1
            else:
                return baryCmp

        # sort neighbors
        _nei = sorted(nei, key=functools.cmp_to_key(cmp_neighbors))
        return _nei

    # BFS traversal
    permArr = []
    for s in range(0, 1):
        # set source as s
        edges = _bfs_edges(G, s, sort_neighbors=sort_func)
        perm = [s] + [v for u, v in edges]
        permArr.append(perm)

    return permArr[0]
