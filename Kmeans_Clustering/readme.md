The file Kmeans_clustering.py contains the code to answer question 1 (a and b) 
of Project 1, Linear Algebra and Optimization for Machine Learning.

K-means clustering using standard Euclidean distance and kernelized distance

Run the python file (main function) and the following results are shown:
1. Option 1: Findings the number of clusters (K), which shows:
    The elbow plot (SSE vs. k number of clusters)
    The silhouette plot (silhouette score vs. k number of clusters)
2. Option 2: K-means clustering using Euclidean distance, which shows:
    SSE and silhouette score for k = 4
    The distribution of each feature based on the 4 clusters
    The number of data points in each cluster
    The heatmap of the correlation matrix of features and cluster
    The 3D scatter plot of three features and cluster based on clustering results using the eleven features and the whole data set
3. Option 3: K-means clustering using kernelized distance, which shows:
    The cost function (SSE) vs. different values of the power of gamma for the gaussian kernel
    The distribution of each feature based on the 4 clusters
    The 3D scatter plot of three features and cluster based on clustering results using the eleven features and the whole data set
