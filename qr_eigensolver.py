# -*- coding: utf-8 -*-
"""
Simple implementation of QR algorithm to solve
standard eigenvalue problems. Instead writing QR
decomposition by myself, I used numpy.linalg.qr
Maybe I'll update my own codes of QR decomposition later...

Created on Tue Oct 25 16:29:48 2022
@author: Meng Zhaonan
"""
import numpy as np
from numpy.linalg import qr

def qr_eigensolver(A,tol,max_iter):
    # A: input matrix
    # tol: error tolerance
    # max_iter: max iteration number
    
    n = len(A)
    L = L_partitioning(A)  # A = L + D + U
    Q = np.identity(n)
    
    for i in range(max_iter):
        q,r = np.linalg.qr(A)
        A = np.dot(r,q)
        Q = np.dot(Q,q)
        L = L_partitioning(A)
        
        print(np.linalg.norm(L))
        
        if np.linalg.norm(L) < tol:
            break
        
    if i == max_iter - 1:
        print("Max iterations reached!")
    else:
        eigval = np.diagonal(A)
        return eigval, Q

def L_partitioning(A):
    # A: input matrix
    n = len(A)
    L = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i][j] = A[i][j]
    return L
    
def test():
    A = np.random.random([10,10])
    A = A + A.T
    eigval,eigvec = qr_eigensolver(A, 0.001, 5000)    
    
test()
    