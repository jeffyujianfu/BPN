import numpy as np


def sinkhorn_knopp(matrix, max_iter=1000, tol=1e-9):
    '''The sinkhorn knopp algorith of generating a n by n doubly stochastic matrix'''
    A = matrix.copy()
    for _ in range(max_iter):
        A /= A.sum(axis=1, keepdims=True)  # Normalize rows
        A /= A.sum(axis=0, keepdims=True)  # Normalize columns
        if np.allclose(A.sum(axis=1), 1, atol=tol) and np.allclose(A.sum(axis=0), 1, atol=tol):
            break
    return A


def social_network(n, k):
    '''generate a n agent social network using the sinkhorn knopp algorithm,
    with each person only connected with k ohters, then save it to a txt file.'''
    if k > n:
        raise ValueError("m cannot be greater than n")
    # generate a n by n triangular matrix with each row having k/n probability of having a non-zero elements
    A = np.zeros((n, n))
    # for each row, choose k of them to be 1
    for i in range(n):
        for j in range(i + 1):  # Explicit index-based iteration
            if np.random.random() < k / n:
                A[i, j] = 1
                
    # make the matrix symmetric and doubly stochastic
    A = A + A.T
    
    # ensure that there is no self edges and that there is at least one edge for each node
    for i in range(n):
        A[i][i] = 0
        if A[i].sum() == 0 or A[:,i].sum() == 0:
            z = np.random.randint(n)
            while z == i:
                z = np.random.randint(n)
            A[i][z] = 1
            A[z][i] = 1
            
    # make it doubly stochastic
    S = sinkhorn_knopp(A)
    
    # save the file
    np.savetxt("data/social_network.txt", S, delimiter=" ", fmt="%.4f")


def generate_random_network(n, k):
    '''generate a n agent social network using the sinkhorn knopp algorithm'''
    # generate a n by n triangular matrix with each row having k/n probability of having a non-zero elements
    A = np.zeros((n, n))
    # for each row, choose k of them to be 1
    for i in range(n):
        for j in range(i + 1):  # Explicit index-based iteration
            if np.random.random() < k / n:
                A[i, j] = 1
                
    # make the matrix symmetric and doubly stochastic
    A = A + A.T
    
    # ensure that there is no self edges and that there is at least one edge for each node
    for i in range(n):
        A[i][i] = 0
        if A[i].sum() == 0 or A[:,i].sum() == 0:
            z = np.random.randint(n)
            while z == i:
                z = np.random.randint(n)
            A[i][z] = 1
            A[z][i] = 1
    return A


def social_network_with_communities(n, k):
    '''generate a n agent social network with 2 communities and c interconnections 
    between the two communities. Using the sinkhorn knopp algorithm, with each person 
    connected to k ohters, then save it to a txt file.'''
    if k > n:
        raise ValueError("m cannot be greater than n")
    m = int(n/2)
    
    # generate 2 communities with m agents each
    A = generate_random_network(m, k)
    B = generate_random_network(m, k)
    
    connection = np.zeros((m, m))
    connection = np.ravel(connection)
    connect = np.random.choice(m*m, 10, replace=False)
    connection[connect] = 1
    connection = connection.reshape((m, m))
    
    # Create the block diagonal matrix
    M = np.block([
        [A, connection],
        [connection, B],
        ])
    
    # finally make it doubly stochastic
    S = sinkhorn_knopp(M)
    
    # save the file
    np.savetxt("data/social_network_with_community.txt", S, delimiter=" ", fmt="%.4f")
    
    
    
def structure_of_understanding(n, m):
    '''generate structure of understanding consists of 2m evidence
       (including opposing evidence) for n agents randomly.'''
    
    # generate a random n by m matrix with each row sum to 1
    W = np.random.rand(n, m)
    W /= W.sum(axis=1, keepdims=True)
    
    S = np.zeros((n,2*m))
    for j in range(n):
        for i in range(m):
            if np.random.rand() > 0.5:
                S[j][i] = W[j][i]
            else:
                S[j][i+m] = W[j][i]
    
    # save the structure of understanding to a txt file
    np.savetxt("data/structure_of_understandings.txt", S, delimiter=" ", fmt="%.4f")


def structure_of_understanding_with_community(n, m, p):
    '''generate structure of understanding consists of 2m evidence
       (including opposing evidence) for n agents where half of the 
       agents choose the positive side of the evidence with probability p, 
       and half of the agents choose the negative side of the evidence 
       with probability p.'''
    
    # generate a random n by m matrix with each row sum to 1
    W = np.random.rand(n, m)
    W /= W.sum(axis=1, keepdims=True)
    
    k = int(n/2)
    S_1 = np.zeros((k, 2*m))
    S_2 = np.zeros((k, 2*m))
    
    for j in range(k):
        for i in range(m):
            if np.random.rand() < p:
                S_1[j][i] = W[j][i]
            else:
                S_1[j][i+m] = W[j][i]
    
    for j in range(k):
        for i in range(m):
            if np.random.rand() < 1-p:
                S_2[j][i] = W[j][i]
            else:
                S_2[j][i+m] = W[j][i]
                
    S = np.vstack((S_1, S_2))
    
    # save the structure of understanding to a txt file
    np.savetxt("data/structure_of_understandings_with_community.txt", S, delimiter=" ", fmt="%.4f")


def initial_confidence_level_on_evidence_with_community(n, m):
    '''generate structure of understanding consists of m evidence for n agents 
    where they are separated into two communities. One leans towards the positive 
    side of the evidence, and one leans towards the negative side of the evidence.'''
    k = int(n/2)
    S_1 = np.ones((n, m), dtype=float)*0.2
    S_2 = np.ones((n, m), dtype=float)*0.8
    S = np.vstack((S_1, S_2))
    # save the structure of understanding to a txt file
    np.savetxt("data/initial_confidence_level_on_evidence_with_community.txt", S, delimiter=" ", fmt="%.4f")


def initial_confidence_level_on_evidence(n, m):
    '''generate structure of understanding consists of m evidence for n agents randomly.'''
    S = np.random.rand(n, m)
    # save the structure of understanding to a txt file
    np.savetxt("data/initial_confidence_level_on_evidence.txt", S, delimiter=" ", fmt="%.4f")
    
    
def constant_initial_confidence_level_on_evidence(n, m, c):
    '''generate structure of understanding consists of m evidence for n agents randomly.'''
    S = np.ones((n, m), dtype=float)*c
    # save the structure of understanding to a txt file
    np.savetxt("data/constant_initial_confidence_level_on_evidence.txt", S, delimiter=" ", fmt="%.4f")


def constant_self_confidence_level(n, c):
    '''generate a constant self confidence level c for n agents.'''
    return np.ones(n)*c