import numpy as np 
from scipy.sparse import diags

# QR algorithm
def qr_eigensolver(A,tol,max_iter):
    '''
    A: input matrix
    tol: error tolerance
    max_iter: max iteration number
    '''
    n = len(A)
    L = L_partitioning(A)  # A = L + D + U
    Q = np.identity(n)     # identity matrix 
    
    for i in range(max_iter):
        q,r = np.linalg.qr(A)  # qr decomposition of A
        A = np.dot(r,q)        
        Q = np.dot(Q,q)
        L = L_partitioning(A)  # lower triangular part of A
        
        if np.linalg.norm(L) < tol:   # convergence criterion
            break
        
    if i == max_iter - 1:
        print("Max iterations reached!")
    else:
        eigval = np.diagonal(A)
        return eigval, Q

# Create lower triangular matrix
def L_partitioning(A):
    '''A: input matrix'''
    n = len(A)
    L = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i][j] = A[i][j]
    return L

# Lanczos algorithm
def lanczos(A,r,k,tol=1e-5):
    '''
    A: Hermitian matrix A n*n
    k: number of iterations
    r: nonzero initial vector r0 (of grade d > k)
    tol: tolerance for lanczos iteration
    '''
    n = len(A)                        # length of A
    r_norm = np.sqrt(np.dot(r.T,r))   # norm of r0
    v = np.zeros([n,1])               # v0 = 0
    v = np.c_[v,r/r_norm]             # v1 = r0/r0_norm
    beta = [0]                       
    gamma = []
    
    for i in range(1,k+1):   
        u = np.dot(A,v[:,i]) - beta[i-1] * v[:,i-1]  # u_k = A*v_k - beta_k-1 * v_k-1
        gamma.append( np.dot(u.T,v[:,i]) )           # gamma_k = (u_k,v_k)
        v = np.c_[v,u - gamma[i-1] * v[:,i]]         # v_k+1 = u_k - gamma_k * v_k
        beta.append( np.sqrt(np.dot(v[:,i+1].T,v[:,i+1])) ) # beta_k = norm of v_k+1
        
        if beta[i] < tol: 
            v[:,i+1] = 0       # when i reaches the grade, we need to break
            break
        else:
            v[:,i+1] = v[:,i+1] / beta[i]   # v_k+1 = v_k+1 / beta_k 
    
    beta = beta[1:]
    T = diags([gamma,beta,beta],[0,1,-1])
    return v[:,1:-1],T

def eigensolver(L):
    '''
    L: input matrix (requires symmetric matrix like laplacian matrix)
    '''
    n = len(L)
    r = np.random.random(n)  # initial vector for lanczos algorithm
    V,T = lanczos(L, r,7)    # 7 is an optimal choice
    
    eigval, eigvec = qr_eigensolver(T.toarray(), 1e-6, 50000)  # apply qr_eigensolver to solver the smaller eigenvalue problem  
    eigvec = V @ eigvec          # basis vector * eigen vector of T
    
    indx = eigval.argsort()      # sort
    eigval = eigval[indx]        # sort
    eigvec = eigvec[:,indx]      # sort
    
    return eigval,eigvec 





