# -*- coding: utf-8 -*-
"""
Spectral Clustering (a)
Created on Wed Nov  9 11:12:10 2022
@author: Zhaonan Meng, TU Delft
"""
import time                                    # timer
import numpy as np                             # numpy 
import csv                                     # read csv
import seaborn as sns                          # heatmap
import matplotlib.pyplot as plt                # visualization 
from sklearn.datasets import make_circles      # generate test circle data
from sklearn.neighbors import kneighbors_graph # n neighbor for adjcency matrix

'''
Import our own libraries
LanczosSolver and Kmeans_eucl are our own algorithms
'''
import sys
import os
sys.path.append(os.getcwd())
from LanczosSolver import eigensolver
from Kmeans_eucl import Kmeans_eucl

# Load input data
def load_data(path):
    try:
        with open(path,"r") as csvfile:
            arr = []
            csvreader = csv.reader(csvfile,delimiter=",")
            next(csvreader)
            for row in csvreader:
                arr.append(row)
        return np.array(arr)[1:]
    except:
        print("File not found!")
        return None
    
# Preprocess data
def data_preprocessing(data,N,f):
    '''
    data: input pandas dataframe 
    N: number of data points you want to use. -1 stands for all data
    f: a list specifying which columns(features) you want to use
    '''
    data = np.array(data[0:N,f])    # which features and how much data do you want
    data = np.float64(data)         # transform data type to float
    #std = np.std(data,axis=0)      # standard deviation of data
    #mean = np.mean(data,axis=0)    # mean value of data
    #data = ( data - mean )/std     # standardization
    data = (data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))   # min max scaling
    return data

# Construct the kernel matrix
def kernel_matrix(data,kernel):
    N = data.shape[0]              # number of data points
    kernel_mat = np.zeros([N,N])   # construct the kernel matrix k[i,j]
    for i in range(N):
        for j in range(i+1):
            kernel_mat[i,j] = kernel(data[i], data[j]) # symmetric matrix saves us from computing every kernel
    kernel_mat = kernel_mat + kernel_mat.T - np.diag(np.diag(kernel_mat))  
    return kernel_mat

# Construct the laplacian matrix
def Laplacian_matrix(data,kernel,nor=False):            # construct a laplacian matrix and check the eigen pairs
    '''
    data: input data
    kernel: select a kernel for the similarity matrix
    nor: whether to use the normalized spectral clustering (False by default)
    '''
    # Gaussian kernel
    def gaussian(x1,x2,gamma=1):                        # by default gamma=1
        return np.exp(-gamma * np.dot(x1-x2,x1-x2))
    # Polynomial kernel
    def polynomial(x1,x2,d=2,c=0):                      # by default d=2 and c=0
        return (np.dot(x1,x2) + c) ** d

    stime = time.time()
    
    # kernel matrix W for all data
    if kernel == "gaussian":    
        W = kernel_matrix(data, gaussian)   
    elif kernel == "polynomial":
        W = kernel_matrix(data, polynomial)
    else:
        print("No matching kernel!")
        return None
    
    if nor == False: 
        D = np.diag(W.sum(-1))               # D_{ii} = /sum_{j=1}^N w_{ij}
        L = D - W                            # unnormalized laplacian matrix
    elif nor == True:
        D = np.diag(1 / np.sqrt(W.sum(-1)))  # D^-0.5
        L = np.identity(len(D)) - D @ W @ D  # normalized laplacian matrix
    else:
        print("actual argument for nor is incorrect!")
        return None
    
    etime = time.time()
    print("kernel matrix construction completed, taking {} s".format(etime - stime))
    return L

# Use numpy.linalg.eig to compute eigenvalues and eigenvectors and sort them
def numpy_eigensolver(m,n):
    '''
    m: input matrix
    n: number of smallest eigen pairs you want
    '''
    eigval,eigvec = np.linalg.eigh(m)   # compute eigen pairs
    indx = eigval.argsort()             # sort
    eigval = eigval[indx]               # sort
    eigvec = eigvec[:,indx]             # sort
    return eigval[0:n],eigvec[:,0:n]

# Apply K-means clustering to rows of eigenvectors
def pseudo_ratiocut(H,max_iter):
    '''
    H: input matrix formed by eigenvectors
    max_iter: max number of iterations for Euclidean K-means clustering 
    '''
    k = H.shape[1]
    m,w,ESS = Kmeans_eucl(H, k, max_iter)  # Apply Euclidean K-means clustering to rows of eigenvector matrix 
    return m,w,ESS     # m: clustering matrix  w: centroid position ESS: cost_function

# Generating another laplacian matrix by the adjacency matrix
def generate_another_graph_laplacian(df, nn):
    '''
    This function is for sec2_1, where Laplacian matrix is constructed
    by nearest k neighbors instead of kernel function.
    '''
    connectivity = kneighbors_graph(X=df, n_neighbors=nn, mode='connectivity')
    adjacency_matrix_s = (1/2)*(connectivity + connectivity.T).toarray()  # adjacency matrix
    D = np.diag(adjacency_matrix_s.sum(-1))                               # diagonal matrix
    graph_laplacian = D - adjacency_matrix_s             
    return graph_laplacian 

# Generate some circle data (3 circles)
def generate_circdata():
    '''
    generate some circle data for the purpose of correctness testing
    '''
    X_small, y_small = make_circles(n_samples=(250,500), random_state=3, noise=0.04, factor = 0.3)
    X_large, y_large = make_circles(n_samples=(250,500), random_state=3, noise=0.04, factor = 0.7)
    X = np.r_[X_large,X_small]
    return X
    
# Test our spectral clustering   
def sec2_1():
    '''
    section 2.1: Correctness Test

    '''
    data,label = make_circles(300,noise=0.03,factor=0.5)    # make some circle data
    
    fig,ax = plt.subplots(figsize=[10,8])
    ax.scatter(data[:,0],data[:,1],s=5)
    plt.show()
    '''
    For circle data clustering I don't use kernel laplacian matrix.
    Instead, I use the laplacian matrix constructed by adjecency matrix.
    '''
    L = generate_another_graph_laplacian(df = data, nn=8)
    
    eigval,eigvec = numpy_eigensolver(L, 2)    # numpy.linalg.eigh(...)
    
    fig, ax = plt.subplots(figsize=(10, 8))    # heatmap for laplacian matrices
    sns.heatmap(eigvec.real, ax=ax, cmap='viridis_r')
    ax.set(title='Eigenvectors Generating the Kernel of the Graph Laplacian');
    plt.show()
    
    m,w,ESS = Kmeans_eucl(eigvec, 2, 20)       # Euclidean K-Means clustering
    
    fig,ax = plt.subplots()
    data0 = data[np.where(m[0]==1)]
    data1 = data[np.where(m[1]==1)]
    ax.scatter(data0.T[0],data0.T[1],s=5,c='mediumslateblue',label='class 1')
    ax.scatter(data1.T[0],data1.T[1],s=5,c='tomato',label='class 2')
    ax.legend()
    ax.grid()
    plt.show()
    
    fig,ax = plt.subplots()
    ax.scatter(eigvec.T[0],eigvec.T[1],c='blue',label='eigenvector dataset')
    ax.scatter(w.T[0],w.T[1],c='red',label='centroids')
    ax.legend()
    plt.show()
    
def sec2_2():
    '''
    section 2.2: Spectral Clustering with numpy.linalg.eigh
    '''
    # Loading and proprocessing data
    df   = load_data(r'.\EastWestAirlinesCluster.csv')
    # -1 stands for all data. [1,3,4,9] represents [balance,cc1,cc2,flight trans]
    data = data_preprocessing(df, -1, [1,3,4,9])   
    
    # Construct normalized and unnormalized laplacian matrix
    L_unnorm = Laplacian_matrix(data, 'gaussian',False)
    L_norm = Laplacian_matrix(data, 'gaussian',True)
    
    # Visualizing laplacian matrices
    plt.figure(figsize=[10,4])
    plt.subplot(121)
    sns.heatmap(L_unnorm, cmap='viridis_r')  # unnormalized laplacian matrix
    plt.subplot(122)                         
    sns.heatmap(L_norm, cmap='viridis_r')    # normalized laplacian matrix
    plt.show()
    
    # Compute 5 smallest eigenvalues and the corresponding eigenvectors
    eigval_ln,eigvec_ln = numpy_eigensolver(L_norm, 5)   # normalized laplacian matrix
    eigval_lu,eigvec_lu = numpy_eigensolver(L_unnorm, 5) # unnormalized laplacian matrix
    fig, ax = plt.subplots(1,2,figsize=(12, 4))
    ax[0].plot(eigval_lu,c='black',linewidth=1)
    ax[1].plot(eigval_ln,c='black',linewidth=1)
    ax[0].scatter([0,1,2,3],eigval_lu[0:4],c='red')
    ax[1].scatter([0,1,2,3],eigval_ln[0:4],c='red')
    ax[0].set_ylabel('eigenvalue')
    ax[0].set_xlabel('index')
    ax[1].set_xlabel('index')
    ax[0].grid(),ax[1].grid()
    plt.show()
    
    # Applying Euclidean K-means (ratiocut)
    m_n,w_n,ESS_n = pseudo_ratiocut(eigvec_ln[:,0:4], 20)  # normalized spectral clustering
    m_u,w_u,ESS_u = pseudo_ratiocut(eigvec_lu[:,0:4], 20)  # unnormalized spectral clustering
    
    # check the convergence of EES
    fig, ax = plt.subplots(1,2,figsize=(12, 4))
    ax[0].plot(ESS_u,c='black',linewidth=1)
    ax[1].plot(ESS_n,c='black',linewidth=1)
    ax[0].set_xlabel('iteration')
    ax[1].set_xlabel('iteration')
    ax[0].set_ylabel('ESS')
    ax[0].grid(),ax[1].grid()
    plt.show()
    
    '''
    Clustering data by applying K-means to eigenvectors of the 
    unnormalized laplacian matrix
    '''
    data0_u = data[np.where(m_u[0]==1)]  # cluster 0
    data1_u = data[np.where(m_u[1]==1)]  # cluster 1
    data2_u = data[np.where(m_u[2]==1)]  # cluster 2
    data3_u = data[np.where(m_u[3]==1)]  # cluster 3
    
    fig,ax = plt.subplots(4,4,sharey=True,figsize=[15,13])
    for i in range(4):
        for j in range(4):
            ax[i,j].scatter(data0_u.T[j],data0_u.T[i],s=7,c='royalblue')
            ax[i,j].scatter(data1_u.T[j],data1_u.T[i],s=7,c='limegreen')
            ax[i,j].scatter(data2_u.T[j],data2_u.T[i],s=7,c='firebrick')
            ax[i,j].scatter(data3_u.T[j],data3_u.T[i],s=7,c='yellow')
            ax[i,j].grid()
    ax[0,0].set_ylabel('balance')
    ax[1,0].set_ylabel('cc1 miles')
    ax[2,0].set_ylabel('cc2 miles')
    ax[3,0].set_ylabel('flight trans 12')
    ax[0,0].set_title('balance')
    ax[0,1].set_title('cc1 miles')
    ax[0,2].set_title('cc2 miles')
    ax[0,3].set_title('flight trans 12')
    plt.show()
    '''
    Clustering data by applying K-means to eigenvectors of the 
    normalized laplacian matrix
    '''
    data0_n = data[np.where(m_n[0]==1)]  # cluster 0
    data1_n = data[np.where(m_n[1]==1)]  # cluster 1
    data2_n = data[np.where(m_n[2]==1)]  # cluster 2
    data3_n = data[np.where(m_n[3]==1)]  # cluster 3
    
    fig,ax = plt.subplots(4,4,sharey=True,figsize=[15,13])
    for i in range(4):
        for j in range(4):
            ax[i,j].scatter(data0_n.T[j],data0_n.T[i],s=7,c='royalblue')
            ax[i,j].scatter(data1_n.T[j],data1_n.T[i],s=7,c='limegreen')
            ax[i,j].scatter(data2_n.T[j],data2_n.T[i],s=7,c='firebrick')
            ax[i,j].scatter(data3_n.T[j],data3_n.T[i],s=7,c='yellow')
            ax[i,j].grid()
    ax[0,0].set_ylabel('balance')
    ax[1,0].set_ylabel('cc1 miles')
    ax[2,0].set_ylabel('cc2 miles')
    ax[3,0].set_ylabel('flight trans 12')
    ax[0,0].set_title('balance')
    ax[0,1].set_title('cc1 miles')
    ax[0,2].set_title('cc2 miles')
    ax[0,3].set_title('flight trans 12')
    plt.show()
    
def sec2_3():
    '''
    section 2.3: Spectral Clustering with lanczos eigensolver
    '''
    # Loading and proprocessing data
    df   = load_data(r'.\EastWestAirlinesCluster.csv')
    # -1 stands for all data. [1,3,4,9] represents [balance,cc1,cc2,flight trans]
    data = data_preprocessing(df, -1, [1,3,4,9])
    
    # Construct normalized and unnormalized laplacian matrix
    L_unnorm = Laplacian_matrix(data, 'gaussian',False)
    L_norm = Laplacian_matrix(data, 'gaussian',True)
    
    # Visualizing laplacian matrices
    plt.figure(figsize=[10,4])
    plt.subplot(121)
    sns.heatmap(L_unnorm, cmap='viridis_r')  # unnormalized laplacian matrix
    plt.subplot(122)                         
    sns.heatmap(L_norm, cmap='viridis_r')    # normalized laplacian matrix
    plt.show()
    
    # Compute 5 smallest eigenvalues and the corresponding eigenvectors
    eigval_ln,eigvec_ln = eigensolver(L_norm)   # normalized laplacian matrix
    eigval_lu,eigvec_lu = eigensolver(L_unnorm) # unnormalized laplacian matrix
    
    fig, ax = plt.subplots(1,2,figsize=(12, 4))
    ax[0].plot(eigval_lu,c='black',linewidth=1)
    ax[1].plot(eigval_ln,c='black',linewidth=1)
    ax[0].scatter([0,1,2,3],eigval_lu[0:4],c='red')
    ax[1].scatter([0,1,2,3],eigval_ln[0:4],c='red')
    ax[0].set_ylabel('eigenvalue')
    ax[0].set_xlabel('index')
    ax[1].set_xlabel('index')
    ax[0].grid(),ax[1].grid()
    plt.show()
    
    # Applying Euclidean K-means (ratiocut)
    m_n,w_n,ESS_n = pseudo_ratiocut(eigvec_ln[:,0:4], 20)
    m_u,w_u,ESS_u = pseudo_ratiocut(eigvec_lu[:,0:4], 20)
    
    # check the convergence of EES
    fig, ax = plt.subplots(1,2,figsize=(12, 4))
    ax[0].plot(ESS_u,c='black',linewidth=1)
    ax[1].plot(ESS_n,c='black',linewidth=1)
    ax[0].set_xlabel('iteration')
    ax[1].set_xlabel('iteration')
    ax[0].set_ylabel('ESS')
    ax[0].grid(),ax[1].grid()
    plt.show()
    '''
    Clustering data by applying K-means to eigenvectors of the 
    unnormalized laplacian matrix
    '''
    data0_u = data[np.where(m_u[0]==1)]  # cluster 0
    data1_u = data[np.where(m_u[1]==1)]  # cluster 1
    data2_u = data[np.where(m_u[2]==1)]  # cluster 2
    data3_u = data[np.where(m_u[3]==1)]  # cluster 3
    fig,ax = plt.subplots(4,4,sharey=True,figsize=[15,13])
    for i in range(4):
        for j in range(4):
            ax[i,j].scatter(data0_u.T[j],data0_u.T[i],s=7,c='royalblue')
            ax[i,j].scatter(data1_u.T[j],data1_u.T[i],s=7,c='limegreen')
            ax[i,j].scatter(data2_u.T[j],data2_u.T[i],s=7,c='firebrick')
            ax[i,j].scatter(data3_u.T[j],data3_u.T[i],s=7,c='yellow')
            ax[i,j].grid()
    ax[0,0].set_ylabel('balance')
    ax[1,0].set_ylabel('cc1 miles')
    ax[2,0].set_ylabel('cc2 miles')
    ax[3,0].set_ylabel('flight trans 12')
    ax[0,0].set_title('balance')
    ax[0,1].set_title('cc1 miles')
    ax[0,2].set_title('cc2 miles')
    ax[0,3].set_title('flight trans 12')
    plt.show()
    '''
    Clustering data by applying K-means to eigenvectors of the 
    normalized laplacian matrix
    '''
    data0_n = data[np.where(m_n[0]==1)]  # cluster 0
    data1_n = data[np.where(m_n[1]==1)]  # cluster 1
    data2_n = data[np.where(m_n[2]==1)]  # cluster 2
    data3_n = data[np.where(m_n[3]==1)]  # cluster 3
    fig,ax = plt.subplots(4,4,sharey=True,figsize=[15,13])
    for i in range(4):
        for j in range(4):
            ax[i,j].scatter(data0_n.T[j],data0_n.T[i],s=7,c='royalblue')
            ax[i,j].scatter(data1_n.T[j],data1_n.T[i],s=7,c='limegreen')
            ax[i,j].scatter(data2_n.T[j],data2_n.T[i],s=7,c='firebrick')
            ax[i,j].scatter(data3_n.T[j],data3_n.T[i],s=7,c='yellow')
            ax[i,j].grid()
    ax[0,0].set_ylabel('balance')
    ax[1,0].set_ylabel('cc1 miles')
    ax[2,0].set_ylabel('cc2 miles')
    ax[3,0].set_ylabel('flight trans 12')
    ax[0,0].set_title('balance')
    ax[0,1].set_title('cc1 miles')
    ax[0,2].set_title('cc2 miles')
    ax[0,3].set_title('flight trans 12')
    plt.show()
    
def eigensolver_test():
    '''
    additional section: comparison between numpy.linalg.eigh and Lanczos solver
    '''
    df   = load_data(r'.\EastWestAirlinesCluster.csv') 
    data = data_preprocessing(df,-1,[1,3,4,9])       # We use all data with 4 features: "balance", "cc1_miles", "cc2_miles", "Flight_trans_12"
    L = Laplacian_matrix(data, 'gaussian',True)      # Laplacian matrix built by gaussian kernel

    stime = time.time()                              # Starting time 
    eigval_np, eigvec_np = numpy_eigensolver(L, 7)   # Using numpy.linalg.eig
    etime = time.time()                              # Ending time
    
    print("Numpy.linalg.eig complete, taking {} seconds".format(etime-stime))
    print("5 smallest eigenvalue computed by linalg.eigh: ",eigval_np[0:7])
    print("And the corresponding 5 eigenvectors: ",eigvec_np[:,0:7])
    
    stime = time.time()                              # Starting time
    eigval_ls, eigvec_ls = eigensolver(L)            # Using our own Lanczos eigensolver
    etime = time.time()                              # Ending time
    
    print("Our own lanczos eigensolver complete, taking {} seconds".format(etime-stime))
    print("5 smallest eigenvalue computed by lanczos solver: ",eigval_ls[0:7])
    print("And the corresponding 5 eigenvectors: ",eigvec_ls[:,0:7])


def main():
    print("Hi! Welcome to spectral_clustering.py! This program is a part of the\
          project 1 of the course <Linear Algebra and Optimization for Machine \
          Learning>, implementing spectral clustering using different laplacian\
          matrices and eigensolvers. Please select what you want to see:\n\
          [1]Section 2.1: Correctness Test\n\
          [2]Section 2.2: Spectral Clustering with numpy.linalg.eig\n\
          [3]Section 2.3: Spectral Clustering with lanczos eigensolver\n\
          [4]Addition: Efficiency and Effectiveness for the eigensolver")
    
    while(True):
        print("Please make your choice [1-4]:")
        x = input()
        
        if x == "1":
            print("Please wait a few seconds. Running sec2_1() will take several seconds.")
            sec2_1()    
        elif x == "2":
            print("Please wait about 3 minutes. Running sec2_2() will take around 3 minutes. ")
            sec2_2()      
        elif x == "3":
            print("Please wait about 2 minutes. Running sec2_3() will take around 2 minutes. ")
            sec2_3()
        elif x == "4":
            print("Please wait about 1 minute. Running eigensolver_test() will take around 1 minute. ")
            eigensolver_test()
        else:
            print("Your input is incorrect. Please choose from 1 to 4")
            
        print("Do you want to continue? (y/n)")
        while(True):
            y = input()
            if y == 'y':
                break
            elif y == 'n':
                return None
            else:
                print("Your input is incorrect. Please select y or n:")
      
main()


