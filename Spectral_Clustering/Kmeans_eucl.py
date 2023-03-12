# -*- coding: utf-8 -*-
"""
K-means Clustering 
Euclidean distance

Created on Tue Oct 25 17:34:59 2022
@author: Meng Zhaonan
"""
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

# Euclidean K-means clustering
def Kmeans_eucl(data,k,max_iter):
    '''
    data: input data matrix N*n (N: # of data n: # of features)
    k: number of centroids
    max_iter: max iteration number
    '''
    N = data.shape[0]            # number of data points
    n = data.shape[1]            # number of features
    
    w = np.zeros([k,n])                    # initialize position of centroids
    cmean_of_data = data.mean(0)           # column mean of input data
    
    for i in range(k):
        '''
        Here we initialize our centroids. The tricky thing is that we need to pay much
        attention if the initial centroids are too far away from the whole dataset.
        Sometimes one may be not able to find a correct clustering if the initial centroids
        are too far. Therefore, I apply some scaling here, to make sure that the initial
        centroids are close to the whole dataset.
        '''
        w[i] = cmean_of_data + cmean_of_data * np.random.rand(n) - cmean_of_data/2.0
        
    ESS = []                               # list of error sum of squares    
    old_cost = 0
    for i in range(max_iter):              # Start clustering
        # iterate over the prototype matrix to find the data belonging to the prototype
        # i.e. construct the m_q matrix   
        m = np.zeros([k,N])                # initialize m_q matrix
        for j in range(N): 
            dis = 0                        # compute the distance
            for l in range(n):
                d = w[:,l] - data[j,l]
                dis += d * d
            dis = np.sqrt(dis)             # distance = sqrt(d1^2+d2^2...)
            m[np.argmin(dis),j]=1          # assign the data point to the nearest centroid
            '''
            Note that: Sometimes one prototype may not possess any data point. 
            In this case, we need to re-initialize the position of this prototype 
            since we cannot update its position by setting w_q to the center 
            of mass of its assigned data.
            '''   
        cost = 0                           # Cost 
        for p in range(k):                 # update centroids
            cost += np.dot(m[p], np.sum((data - w[p]) * (data - w[p]), 1) )
            if np.sum(m[p]) == 0:
                w[p] = cmean_of_data + cmean_of_data * np.random.rand(n) - cmean_of_data/2.0
            else:
                w[p] = np.dot(m[p],data)/np.sum(m[p])  # update centroid position
        ESS.append(cost)
        
        if abs(old_cost - cost) < 1e-5:               # if no improvement then break
            return m,w,ESS
        else:
            old_cost = cost
    
    print("Warning: max iterations reached! Please check the convergence.")
    return m,w,ESS


#def euclidean_test():
#    df=pd.read_csv(r"C:\Users\qqsup\Desktop\COSSE\2022,23WS\linear algebra and optimization for machine learning\cluster.csv", engine='python')
#    data = np.array(df)
#    w = Kmeans_Eucl(data, 4,20)
#    plt.scatter(data.T[0],data.T[1],c='lightslategrey',s=7)
#    plt.scatter(w.T[0],w.T[1],c='r',s=10)
#    
#euclidean_test()
