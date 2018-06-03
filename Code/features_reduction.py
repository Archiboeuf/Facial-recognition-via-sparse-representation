from time import time
import logging
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os,sys
import numpy as np
from sklearn.model_selection import train_test_split

## EigenFace
## Fisherfaces
## RandomFaces

def eigenfaces(X_train,X_test,n_components=1260):
    
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


    print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver='randomized',
      whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))
    
    
    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("EigenFaces done in %0.3fs" % (time() - t0))
    
    
    return X_train_pca, X_test_pca
    
    
def fisherfaces(X_train,y_train,X_test,n_components=1260):
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


    print("Extracting the top %d Fisherfaces from %d faces"
      % (n_components, X_train.shape[0]))
    t0 = time()
    lda = LDA(n_components=n_components).fit(X_train,y_train)
    print("done in %0.3fs" % (time() - t0))
    
    
    print("Projecting the input data on the fisher orthonormal basis")
    t0 = time()
    X_train_fisher = lda.transform(X_train)
    X_test_fisher = lda.transform(X_test)
    print("Fisherfaces done in %0.3fs" % (time() - t0))
    return X_train_fisher,X_test_fisher
    
    
def randomfaces(X_train,X_test, n_components=1260):
    t0 = time()
    randomface = GaussianRandomProjection(n_components=n_components) #Gaussian projection
    randomface.fit(X_train)
    X_train_random = randomface.transform(X_train)
    X_test_random = randomface.transform(X_test)
    print("Random projection done in %0.3fs" % (time() - t0))
    return X_train_random, X_test_random

   
if __name__ == "__main__":
    X,y = read_images(path=path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # EigenFaces
    X_train_pca, X_test_pca = eigenfaces(X_train,X_test,n_components=1260)
    
    # FisherFaces
    X_train_fisher,X_test_fisher= fisherfaces(X_train,y_train,X_test,n_components=1260)
    
    #RandomFaces
    X_train_random, X_test_random = randomfaces(X_train,X_test)
    
    
