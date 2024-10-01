"""Implementation of methods in 
Lin, Zhenhua.
Riemannian geometry of symmetric positive definite matrices via Cholesky decomposition. 
SIAM Journal on Matrix Analysis and Applications 40, no. 4 (2019): 1353-1370.

We also implement supervised minimum distance to mean (MDM) on the manifold and the 
unsupervised k-medoids clustering on the manifold.
"""

import numpy as np
from sklearn.manifold import TSNE
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import cdist

def splitMatrix(matrix , LOG):
    
    strictlyLowerTriangularMatrix = np.tril(matrix, k = -1)
    if LOG == True:
        diagonalMatrix = np.diag(np.log(np.diag(matrix)))
    else:
        diagonalMatrix = np.diag(np.diag(matrix))
    
    return diagonalMatrix, strictlyLowerTriangularMatrix

class matrixDistance:
    def __init__(self):
        self.LOG = True

    def distance(self, matrix1, matrix2):
        chol1 = np.linalg.cholesky(matrix1)
        chol2 = np.linalg.cholesky(matrix2)
        chol1D, chol1L = splitMatrix(chol1, self.LOG)
        chol2D, chol2L = splitMatrix(chol2, self.LOG)
        distanceL = np.square(np.linalg.norm(chol1L - chol2L, 'fro'))
        distanceD = np.square(np.linalg.norm(chol1D - chol2D, 'fro'))
        distance = np.sqrt(distanceL + distanceD)
        return distance

class tSNEmbedding:

    def __init__(self):
        self.matrixDistance = matrixDistance()    

    def pairwiseDistances(self, matrix):
        numberMatrices, n, _ = np.shape(matrix)
        DISTANCES = np.zeros((numberMatrices, numberMatrices))
        for i in range(numberMatrices):
            for j in range(i, numberMatrices):
                distance = self.matrixDistance.distance(matrix[i], matrix[j])
                DISTANCES[i, j] = distance
                DISTANCES[j, i] = distance
        return DISTANCES
    
    
    def tSNE(self, covarianceMatrix, Perplexity, dimension):
        temporaryMatrix = covarianceMatrix.reshape(-1, dimension, dimension)
        
        pairwiseDistance = self.pairwiseDistances(temporaryMatrix)
        
        tsne = TSNE(metric = 'precomputed', perplexity = Perplexity, learning_rate = 200, early_exaggeration = 4, init = "random")

        out = tsne.fit_transform(pairwiseDistance)
        return out
    

class unsupervised:

    def __init__(self):
        self.LOG = True
        self.matrixDistance = matrixDistance()

    def pairwiseDistances(self, matrixSet1, matrixSet2, dimension):
        matrixSet1 = matrixSet1.reshape(-1, dimension, dimension)
        matrixSet2 = matrixSet2.reshape(-1, dimension, dimension)
        shape1 = matrixSet1.shape[0]
        shape2 = matrixSet2.shape[0]
        PairwiseDistances = np.zeros((shape1, shape2))
        for i in range(shape1):
            for j in range(shape2):
                distance = self.matrixDistance.distance(matrixSet1[i], matrixSet2[j])
                PairwiseDistances[i, j] = distance
        return PairwiseDistances
    
    def kMedoids(self, covarianceMatrices, dimension, numberClusters):

        PairwiseDistances = self.pairwiseDistances(covarianceMatrices, covarianceMatrices, dimension)
        kmedoids = KMedoids(n_clusters = numberClusters, metric = "precomputed", method = 'pam', init = "k-medoids++", max_iter = 1000, random_state = 0)
        kmedoids.fit(PairwiseDistances)
        return kmedoids
    
class frechetMean:
    def __init__(self):
        self.LOG = True

    def mean(self, matrix):
        numberMatrices, n, _ = np.shape(matrix)
        lowerMatrices = np.zeros((numberMatrices, n, n))
        diagonalMatrices = np.zeros((numberMatrices, n, n))
        for i in range(numberMatrices):
            chol = np.linalg.cholesky(matrix[i, :, :])
            cholD, cholL = splitMatrix(chol, self.LOG)
            lowerMatrices[i, :, :] = cholL
            diagonalMatrices[i, :, :] = cholD
        
        meanL = np.mean(lowerMatrices, axis = 0)
        meanD = np.diag(np.exp(np.diag(np.mean(diagonalMatrices, axis = 0))))

        meanF = meanL + meanD
        return meanF @ meanF.T