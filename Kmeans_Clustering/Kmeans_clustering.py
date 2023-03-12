"""
This is the code to answer question 1 (a and b) of Project 1
Linear Algebra and Optimization for Machine Learning
K-means clustering using standard Euclidean distance and kernelized distance

Run this pyhton file (main function) and the following results are shown:
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
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time

# K-means clustering using standard Euclidean distance (for question 1 (a))
def Kmeans_Eucl(data, k, max_iter, random_seed):
    """
    data: input data matrix N*n (N: # of data n: # of features)
    k: number of centroids
    max_iter: max iteration number
    random_seed: random seed for the initialization of centroids
    """
    # set timer
    start_time = time.time()
    N = data.shape[0]  # number of data points
    n = data.shape[1]  # number of features

    w = np.zeros([k, n])  # initialize position of centroids
    cmean_of_data = data.mean(0)  # column mean of input data
    old_cost = 0

    for i in range(k):
        np.random.seed(random_seed)
        w[i] = cmean_of_data + 2 * np.random.rand(n) - 1

    for i in range(max_iter):  # Start clustering
        # iterate over the prototype matrix to find the data belonging to the prototype
        # i.e. construct the m_q matrix
        m = np.zeros([k, N])  # initialize m_q matrix
        for j in range(N):
            dis = 0  # compute the distance
            for l in range(n):
                d = w[:, l] - data[j][l]
                dis += d * d
            dis = np.sqrt(dis)  # distance = sqrt(d1^2+d2^2...)
            m[np.argmin(dis), j] = 1  # assign the data point to the nearest centroid
            """
            Note that: Sometimes one prototype may not possess any data point. 
            In this case, we need to re-initialize the position of this prototype 
            since we cannot update its position by setting w_q to the center 
            of mass of its assigned data.
            """
        cost = 0  # Cost
        for p in range(k):  # update centroids
            cost += np.dot(m[p], np.sum((data - w[p]) * (data - w[p]), 1))
            if np.sum(m[p]) == 0:
                w[p] = cmean_of_data + 2 * np.random.rand(n) - 1
            else:
                w[p] = np.dot(m[p], data) / np.sum(m[p])  # update centroid position

        if abs(old_cost - cost) < 0.001:  # if no improvement then break
            print(
                "Running time using %s clusters: %s seconds"
                % (k, time.time() - start_time)
            )
            return w, m
        else:
            old_cost = cost

    # print the running time
    print("Running time using %s clusters: %s seconds" % (k, time.time() - start_time))

    return w, m


# standard scaling
def standard_scaling(df):
    """
    df: input data frame
    return: standardized data frame (mean = 0, std = 1)
    """
    df_scaled = df.copy()
    for column in df_scaled.columns:
        df_scaled[column] = (df_scaled[column] - df_scaled[column].mean()) / df_scaled[
            column
        ].std()
    return df_scaled


# our own function to calculate the SSE from scratch
def SSE_scratch(data, labels, w):
    """
    input:
    data: input data matrix N*n (N: # of data n: # of features)
    labels: cluster labels
    w: position of centroids

    return: SSE
    """
    sse = 0
    for i in range(len(data)):
        sse += np.linalg.norm(data[i] - w[labels[i]]) ** 2
    return sse


def SSE_scratch_kernel(K, m):
    """
    input:
    data: N*n kernel matrix (N: # of data)
    m: k*N clusters

    return: SSE
    """
    return np.sum(K.diagonal()) - np.sum(
        np.divide(
            np.sum(m * np.matmul(m, K), axis=1) / (np.sum(m, axis=1) ** 2),
            np.sum(m, axis=1),
        )
    )


# our own function to calculate the silhouette score
def silhouette_score_scratch(data, labels, w, k):
    """
    input:
    data: input data matrix N*n (N: # of data n: # of features)
    labels: cluster labels
    w: position of centroids
    k: number of clusters

    return: silhouette score
    """
    # initialize the silhouette score
    silhouette_score = 0
    # iterate over each data point
    for i in range(len(data)):
        # calculate the average distance between data point i and all other data points in the same cluster
        a = np.linalg.norm(data[i] - w[labels[i]])
        # calculate the smallest (min  operator in the code) mean distance of i to all points in any other cluster
        b = np.inf
        for j in range(k):
            if j != labels[i]:
                b = min(b, np.linalg.norm(data[i] - w[j]))
        # calculate the silhouette score for data point i
        silhouette_score += (b - a) / max(a, b)
    # return the average silhouette score
    silhouette_score /= len(data)
    return silhouette_score


# K-means clustering using kernelization (for question 1(b))
def Kmeans_kernel(kernel_function, data, r, random_seed, k, max_iter, verbose):
    """
    input:
    kernel_function: kernel function (polynomial, gaussian)
    data: input data matrix N*F (N: # of data F: # of features)
    r: kernel parameter (for gaussian kernel, r[0] = gamma)
    random_seed: random seed for the initialization of the members of each cluster (c)
    k: number of clusters
    max_iter: max iteration number
    verbose: print the running time (True/False)

    return:
    K: N*N kernel matrix
    c: k*N matrix, each column represents the membership of each data point to each cluster
    cost: SSE
    """
    # set timer
    start = time.time()
    N = data.shape[0]  # number of data points
    F = data.shape[1]  # number of features
    c = np.zeros((k, N))  # initialize clusters
    # set seed for np.random.randint
    np.random.seed(random_seed)
    for i in range(N):
        c[np.random.randint(0, k), i] = 1

    K = np.zeros([N, N])
    if kernel_function == "polynomial":  # for a 3D parabola, r = 0 and d = 2
        K = (np.matmul(data, data.T) + r[0]) ** r[1]
    elif kernel_function == "gaussian":  # the default value for parameter r is 1
        data_sqr = np.sum(data**2, axis=1, keepdims=True)
        distance_sqr = data_sqr + data_sqr.T - 2 * np.matmul(data, data.T)
        K = np.exp(-r[0] * distance_sqr)
    # end timer
    end = time.time()
    if verbose:
        print("Time taken to compute kernel matrix: ", end - start)
    old_cost = 0
    # set timer
    start = time.time()
    for iter in range(max_iter):
        cost = 0

        C = np.sum(c, axis=1)  # |Ck|
        Ct = np.tile(C, (N, 1)).T  # |Ck| -- 1xN vector version

        A1 = -2 * np.matmul(c, K.T) / Ct  # -2/|Ck| * sum_{i in Ck} k(x_i, x_j)
        A2 = np.sum(c * np.matmul(c, K), axis=1) / (
            C**2
        )  # 1/|Ck|^2 x sum_{i in Ck} sum_{j in Ck} k(x_i, x_j)  -- kx1 vector
        A2t = np.tile(A2, (N, 1)).T  # A2 -- kxN vector version

        distance = K.diagonal() + A1 + A2t

        c = np.zeros([k, N])
        c[np.argmin(distance, axis=0), np.arange(N)] = 1
        cost = np.sum(np.min(distance, axis=0))

        if (old_cost - cost < 10 ** (-15)) and old_cost - cost >= 0:
            end = time.time()
            if verbose:
                print("Time taken to converge: ", end - start)
                # print cost
                print("Cost: ", cost)
            return K, c, cost
        else:
            old_cost = cost
    # end timer
    end = time.time()
    if verbose:
        print("Time taken for ", max_iter, " iterations: ", end - start)
        # print cost
        print("Cost: ", cost)
    return K, c, cost


# main function
if __name__ == "__main__":
    # read data
    print("Reading data...")
    df = pd.read_csv(
        r"EastWestAirlinesCluster.csv",
        engine="python",
    )
    print("Data read successfully!")
    # drop the column 'ID#' from df in new dataframe df_drop
    df_drop = df.drop(columns=["ID#"])

    # scaled df_drop using our own function standard_scaling
    df_drop_scaled = standard_scaling(df_drop)

    # convert df_drop_scaled to numpy array
    data = np.array(df_drop_scaled)

    print(
        "This is Kmeans_clustering.py, the code to answer questions 1 (a) and  (b)\n\
          of project 1 of the course Linear Algebra and Optimization for Machine\n \
          Learning, implementing Kmeans clustering using standard squared Euclidean\n\
          distance and kernelized distance. Please select what you want to see:\n\
          [1]Finding number of clusters (K)\n\
          [2]Answer to 1 (a): K-means clustering using standard squared Euclidean distance\n\
          [3]Answer to 1 (b): K-means clustering using kernelized distance\n\
        "
    )

    answer = True
    while answer:
        choice = int(input("Enter your choice: "))

        if choice == 1:
            # Determine the optimal number of clusters using the elbow method
            # calculate the sum of squared errors for each k using our own function SSE_scratch
            print("Showing the elbow plot in progress...")
            sse = {}
            for k in range(1, 10):
                w, m = Kmeans_Eucl(data, k, 20, 43)
                labels = np.argmax(m.T, axis=1)
                sse[k] = SSE_scratch(data, labels, w)

            # plot the sum of squared errors for each k
            plt.figure()
            plt.plot(list(sse.keys()), list(sse.values()))
            plt.xlabel("Number of cluster")
            plt.ylabel("SSE")
            print(
                "Showing the elbow plot successfully! please close the elbow plot to continue..."
            )
            plt.show()

            # Determine the optimal number of clusters using the silhouette method
            # calculate the silhouette score for each k using our own function silhouette_score_scratch
            print("Showing the silhouette plot in progress...")
            silhouette = {}
            for k in range(2, 10):
                w, m = Kmeans_Eucl(data, k, 20, 43)
                labels = np.argmax(m.T, axis=1)
                silhouette[k] = silhouette_score_scratch(data, labels, w, k)
                # silhouette[k] = silhouette_score(df_drop_scaled, labels)

            # plot the silhouette score for each k
            plt.figure()
            plt.plot(list(silhouette.keys()), list(silhouette.values()))
            plt.xlabel("Number of cluster")
            plt.ylabel("Silhouette score")
            # set the font size larger
            print(
                "Showing the silhouette plot successfully! The optimal number of clusters is 4. please close the silhouette plot to continue..."
            )
            plt.show()

            # We get the optimal number of clusters is 4

        elif choice == 2:
            # ----------------- Kmeans with Euclidean distance -----------------
            # get w using our own function Kmeans_Eucl
            print(
                "K-means clustering using standard Euclidean distance and four clusters in progress..."
            )
            w, m = Kmeans_Eucl(
                data, 4, 20, 43
            )  # number of centroids = 4, max iteration = 20, random seed = 43
            labels = np.argmax(m.T, axis=1)
            print("K-means clustering using standard Euclidean distance completed.")

            # calculate the SSE of the clustering result using our own function SSE_scratch
            sse = SSE_scratch(data, labels, w)
            print("The SSE of the clustering result is: ", sse)

            # calculate the silhouette score of the clustering result using our own function silhouette_score_scratch
            silhouette = silhouette_score_scratch(data, labels, w, 4)
            print("The silhouette score of the clustering result is: ", silhouette)

            # Assign Cluster column to df_labeled
            df_labeled = df.assign(Cluster=labels)

            # plot distribution of each column grouped by Cluster using boxplot
            print("Plotting boxplot...")
            plt.figure(figsize=(25, 25))
            plt.rcParams.update(plt.rcParamsDefault)
            for i, column in enumerate(
                df_labeled.drop(columns=["ID#", "Cluster"]).columns
            ):
                plt.subplot(6, 4, i + 1)
                sns.boxplot(data=df_labeled, x="Cluster", y=column)
            print("Boxplot completed. Close the boxplot window to continue.")
            plt.show()

            # plot count of Cluster
            print("Plotting countplot...")
            sns.countplot(data=df_labeled, x="Cluster")
            print("Countplot completed. Close the countplot window to continue.")
            plt.show()

            # heatmap of df_labeled
            print("Plotting heatmap...")
            plt.figure(figsize=(10, 10))
            sns.heatmap(df_labeled.corr(), annot=True, cmap="coolwarm")
            print("Heatmap completed. Close the heatmap window to continue.")
            plt.show()

            print(
                "Do you want to see the scattered plot of three selected features? It might take sometime to load (y/n)"
            )
            answer1b = True
            while answer1b:
                y = input()
                if y == "y":
                    # make a new dataframe with only the columns we want to plot
                    df_plot = df_labeled[["Flight_trans_12", "cc1_miles", "cc2_miles"]]
                    # scale df_plot using standard_scaling
                    df_plot_scaled = standard_scaling(df_plot)
                    # assign Cluster column to df_plot_scaled
                    df_plot_scaled = pd.DataFrame(
                        df_plot_scaled, columns=df_plot.columns
                    ).assign(Cluster=df_labeled["Cluster"])
                    # plot interactive 3D scatter plot of df_plot with different colors of Cluster
                    print("Plotting interactive 3D scatter plot...")
                    fig = px.scatter_3d(
                        df_plot_scaled,
                        x="Flight_trans_12",
                        y="cc1_miles",
                        z="cc2_miles",
                        color="Cluster",
                    )
                    print(
                        "Interactive 3D scatter plot completed. It might take some time to load the interactive 3D scatter plot."
                    )
                    fig.show()
                    answer1b = False
                elif y == "n":
                    answer1b = False
                else:
                    print("Your input is incorrect. Please select y or n:")

            print("All plots for Kmeans_euclidean are complete.")

        elif choice == 3:
            # ------------------------------Kmeans_kernel--------------------------------
            print("========== Kmeans_kernel ==========")

            print(
                "Plotting the SSE of kernelized Kmeans vs x, where gamma = 10^(-x) ..."
            )
            sse = {}
            # calculate sum of squared errors for different values of gamma
            for k in range(11):
                K, m, cost = Kmeans_kernel(
                    "gaussian", data, [10 ** (-k)], 1, 4, 500, False
                )
                labels = np.argmax(m.T, axis=1)
                sse[k] = cost

            # plot the sum of squared errors for each x ( gamma = 10^(-x) )
            plt.figure()
            plt.plot(list(sse.keys()), list(sse.values()))
            plt.xlabel("Parameter $\gamma = 10^{-x}$")
            plt.ylabel("SSE")
            print(
                "Showing the SSE plot successfully! please close the plot to continue..."
            )
            plt.show()

            # get m using Kmeans_kernel
            K, m, cost = Kmeans_kernel("gaussian", data, [0.01], 1, 4, 1000, True)
            labels = np.argmax(m.T, axis=1)
            # Assign Cluster column to df_labeled
            df_labeled = df.assign(Cluster=labels)
            df_labeled.head()
            # plot distribution of each column grouped by Cluster using boxplot
            print("Plotting boxplot...")
            plt.figure(figsize=(20, 20))
            for i, column in enumerate(
                df_labeled.drop(columns=["ID#", "Cluster"]).columns
            ):
                plt.subplot(6, 4, i + 1)
                sns.boxplot(data=df_labeled, x="Cluster", y=column)
            print("Boxplot completed. Close the boxplot window to continue.")
            plt.show()

            print(
                "Do you want to see the scattered plot of three selected features? It might take sometime to load (y/n)"
            )
            answer1b = True
            while answer1b:
                y = input()
                if y == "y":
                    # make a new dataframe with only the columns we want to plot
                    df_plot = df_labeled[["Flight_trans_12", "cc1_miles", "cc2_miles"]]
                    # scale df_plot using standard_scaling
                    df_plot_scaled = standard_scaling(df_plot)
                    # assign Cluster column to df_plot_scaled
                    df_plot_scaled = pd.DataFrame(
                        df_plot_scaled, columns=df_plot.columns
                    ).assign(Cluster=df_labeled["Cluster"])
                    # plot interactive 3D scatter plot of df_plot with different colors of Cluster
                    print("Plotting interactive 3D scatter plot...")
                    fig = px.scatter_3d(
                        df_plot_scaled,
                        x="Flight_trans_12",
                        y="cc1_miles",
                        z="cc2_miles",
                        color="Cluster",
                    )
                    print(
                        "Interactive 3D scatter plot completed. It might take some time to load the interactive 3D scatter plot."
                    )
                    fig.show()
                    answer1b = False
                elif y == "n":
                    answer1b = False
                else:
                    print("Your input is incorrect. Please select y or n:")

            print("All plots completed.")

        else:
            print("Your input is incorrect. Please choose from 1 to 3")

        print("Do you want to continue? (y/n)")
        answer2 = True
        while answer2:
            y = input()
            if y == "y":
                break
            elif y == "n":
                answer = False
                break
            else:
                print("Your input is incorrect. Please select y or n:")
