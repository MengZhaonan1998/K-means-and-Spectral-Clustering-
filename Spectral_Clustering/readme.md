Hi! Welcome to the second part of our project!
This part focuses on implementing spectral clustering using different eigensolvers.
under project1/Spectral_Clustering, you can find following files:
                
	---------------readme.txt--------------
	---------spectral_clustering.py--------
	------------Kmeans_eucl.py-----------
	-----------LanczosSolver.py-----------
	------EastWestAirlinesCluster.csv-----

spectral_clustering.py: main program of the second part. 
		     Please run this program directly. 
		(Please make sure all .py and .csv files are in the same directory!)
		(Please make sure you have numpy,seaborn,matplotlib,sklearn installed in your python!)
		     You have 4 input options:
		     If you input "1", you will see the results of section 2.1 of our report. (if you don't have sklearn installed, this section is not available.)                                    If you input "2", you will see the results of section 2.2 of our report. (running time is around 3 minutes, mostly contributed by creating kernel matrix)
	                     If you input "3", you will see the results of section 2.3 of our report. (running time is around 2 minutes, mostly contributed by creating kernel matrix)
	                     If you input "4", you will see the comparison of two eigensolvers: numpy.linalg.eig and our own lanczos eigensolver. (running time is around 1 minute, mostly contributed by creating kernel matrix)

Kmeans_eucl.py: Our own library of Euclidean K-means clustering. 
	            In spectral_clustering.py we import and use Kmeans_eucl.py to cluster rows of eigenvectors.

LanczosSolver.py: Our own library of Lanczos eigensolver.
	            In spectral_clustering.py we import and use LanczosSolver.py to find smallest eigenpairs approximately.

EastWestAirlinesCluster.csv: data set. In this project we use the whole data set.


Thanks for attention.
