# start.py
# pip install cvxopt
# pip install cvxpy

#Let's import all the library that we need
import os
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from gensim.models import FastText
import cvxopt
from cvxopt import matrix
import cvxpy as cp
import scipy.sparse as sparse
from tqdm import tqdm
import pickle

# 1) Let's import all of the dataset we used, only the sequences dataset

X0_train = pd.read_csv("/Users/godchrist/Downloads/kernel-methods-ammi-2023/Xtr0.csv", sep=",", index_col=0).values
X1_train = pd.read_csv("/Users/godchrist/Downloads/kernel-methods-ammi-2023/Xtr1.csv", sep=",", index_col=0).values
X2_train = pd.read_csv("/Users/godchrist/Downloads/kernel-methods-ammi-2023/Xtr2.csv", sep=",", index_col=0).values

X0_test = pd.read_csv("/Users/godchrist/Downloads/kernel-methods-ammi-2023/Xte0.csv", sep=",", index_col=0).values
X1_test = pd.read_csv("/Users/godchrist/Downloads/kernel-methods-ammi-2023/Xte1.csv", sep=",", index_col=0).values
X2_test = pd.read_csv("/Users/godchrist/Downloads/kernel-methods-ammi-2023/Xte2.csv", sep=",", index_col=0).values

# shape (2000,1): 0 or 1
Y0_train = pd.read_csv("/Users/godchrist/Downloads/kernel-methods-ammi-2023/Ytr0.csv", sep=",", index_col=0).values
Y1_train = pd.read_csv("/Users/godchrist/Downloads/kernel-methods-ammi-2023/Ytr1.csv", sep=",", index_col=0).values
Y2_train = pd.read_csv("/Users/godchrist/Downloads/kernel-methods-ammi-2023/Ytr2.csv", sep=",", index_col=0).values


#Put the train and the test matrices into the right format
X0_train = X0_train[:,0]
X1_train = X1_train[:,0]
X2_train = X2_train[:,0]

X0_test = X0_test[:,0]
X1_test = X1_test[:,0]
X2_test = X2_test[:,0]

#Rescaling labels
Y0_train = np.where(Y0_train == 0, -1, 1)
Y1_train = np.where(Y1_train == 0, -1, 1)
Y2_train = np.where(Y2_train == 0, -1, 1)


# 2) Let's now implement the Support Vector Machine classes
class SVM():
    """
    SVM implementation
    
    Usage:
        svm = SVM(kernel='linear', C=1)
        svm.fit(X_train, y_train)
        svm.predict(X_test)
    """

    def __init__(self, kernel, C=1.0, tol_support_vectors=1e-4):
        """
        kernel: Which kernel to use
        C: float > 0, default=1.0, regularization parameter
        tol_support_vectors: Threshold for alpha value to consider vectors as support vectors
        """
        self.kernel = kernel
        self.C = C
        self.tol_support_vectors = tol_support_vectors

    def fit(self, X, y):

        self.X_train = X
        n_samples = X.shape[0]
        print("Computing the kernel...")
        self.X_train_gram = self.kernel.gram(X)
        print("Done!")

        #Define the optimization problem to solve

        P = self.X_train_gram
        q = -y.astype('float')
        G = np.block([[np.diag(np.squeeze(y).astype('float'))],[-np.diag(np.squeeze(y).astype('float'))]])
        h = np.concatenate((self.C*np.ones(n_samples),np.zeros(n_samples)))

        #Solve the problem
        #With cvxopt

        P=matrix(P)
        q=matrix(q)
        G=matrix(G)
        h=matrix(h)
        solver = cvxopt.solvers.qp(P=P,q=q,G=G,h=h)
        x = solver['x']
        self.alphas = np.squeeze(np.array(x))

        #Retrieve the support vectors
        self.support_vectors_indices = np.squeeze(np.abs(np.array(x))) > self.tol_support_vectors
        self.alphas = self.alphas[self.support_vectors_indices]
        self.support_vectors = self.X_train[self.support_vectors_indices]

        print(len(self.support_vectors), "support vectors out of",len(self.X_train), "training samples")

        return self.alphas


    def predict(self, X):
        """
        X: array (n_samples, n_features)\\
        Return: float array (n_samples,)
        """
        K = self.kernel.gram(X, self.support_vectors)
        y = np.dot(K, self.alphas)
        return y

    def predict_classes(self, X, threshold=0):
        """
        X: array (n_samples, n_features)\\
        Return: 0 and 1 array (n_samples,)
        """
        K = self.kernel.gram(X, self.support_vectors)
        y = np.dot(K, self.alphas)
        return np.where(y > threshold, 1, -1)


# 3) Implementation of the kernel that we used for the best score: Mismatch and Sum of Mismatch kernel

class Kernel():
    """ Abstract Kernel class"""

    def __init__(self):
        pass

    def similarity(self, x, y):
        """ Similarity between 2 feature vectors (depends on the type of kernel)"""
        return -1

    def gram(self, X1, X2=None):
        """ Compute the gram matrix of a data vector X where the (i,j) entry is defined as <Xi,Xj>\\
        X1: data vector (n_samples_1 x n_features)
        X2: data vector (n_samples_2 x n_features), if None compute the gram matrix for (X1,X1)
        """
        if X2 is None: 
            X2=X1
        n_samples_1 = X1.shape[0]
        n_samples_2 = X2.shape[0]
        G = np.zeros((n_samples_1, n_samples_2))
        for ii in tqdm(range(n_samples_1)):
            for jj in range(n_samples_2):
                G[ii,jj] = self.similarity(X1[ii], X2[jj])
        return G


class SumKernel(Kernel):

    def __init__(self, kernels, weights=None):
        """ kernels: list of kernels """
        self.kernels = kernels
        self.weights = weights
        if self.weights is None:
            self.weights = [1.0 for _ in kernels]
        super().__init__()

    def similarity(self, x, y):
        """ x, y: string """
        s = self.kernels[0].similarity(x,y) * self.weights[0]
        for ii, kernel in enumerate(self.kernels[1:]):
            s += kernel.similarity(x,y) * self.weights[ii]
        return s

    def gram(self, X1, X2=None):
        """ Compute the sum of the gram matrices of all kernels\\
        X1: array of string (n_samples_1,)
        X2: array of string (n_samples_2,), if None compute the gram matrix for (X1,X1)
        """
        G = self.kernels[0].gram(X1,X2) * self.weights[0]
        for ii, kernel in tqdm(enumerate(self.kernels[1:])):
            G += kernel.gram(X1,X2) * self.weights[ii]
        return G


class MismatchKernel(Kernel):

    def __init__(self, k, m, neighbours, kmer_set, normalize=False):
        super().__init__()
        self.k = k
        self.m = m
        self.kmer_set = kmer_set 
        self.neighbours = neighbours
        self.normalize = normalize

    def neighbour_embed_kmer(self, x):
        """
        Embed kmer with neighbours.
        x: str
        """
        kmer_x = [x[j:j + self.k] for j in range(len(x) - self.k + 1)]
        x_emb = {}
        for kmer in kmer_x:
            neigh_kmer = self.neighbours[kmer]
            for neigh in neigh_kmer:
                idx_neigh = self.kmer_set[neigh]
                if idx_neigh in x_emb:
                    x_emb[idx_neigh] += 1
                else:
                    x_emb[idx_neigh] = 1
        return x_emb
        

    def neighbour_embed_data(self, X):
        """
        Embed data with neighbours.
        X: array of string
        """
        X_emb = []
        for i in range(len(X)):
            x = X[i]
            x_emb = self.neighbour_embed_kmer(x)
            X_emb.append(x_emb)
        return X_emb
    
    def to_sparse(self, X_emb):
        """
        Embed data to sparse matrix.
        X_emb: list of dict.
        """
        data, row, col = [], [], []
        for i in range(len(X_emb)):
            x = X_emb[i]
            data += list(x.values())
            row += list(x.keys())
            col += [i for j in range(len(x))]
        X_sm = sparse.coo_matrix((data, (row, col)))
        return X_sm

    def similarity(self, x, y):
        """ Mismatch kernel \\
        x, y: string
        """
        x_emb = self.neighbour_embed_kmer(x)
        y_emb = self.neighbour_embed_kmer(y)
        sp = 0
        for idx_neigh in x_emb:
            if idx_neigh in y_emb:
                sp += x_emb[idx_neigh] * y_emb[idx_neigh]
        if self.normalize:
            sp /= np.sqrt(np.sum(np.array(list(x_emb.values()))**2))
            sp /= np.sqrt(np.sum(np.array(list(y_emb.values()))**2))
        return sp

    def gram(self, X1, X2=None):
        """ Compute the gram matrix of a data vector X where the (i,j) entry is defined as <Xi,Xj>\\
        X1: array of string (n_samples_1,)
        X2: array of string (n_samples_2,), if None compute the gram matrix for (X1,X1)
        """
        
        X1_emb = self.neighbour_embed_data(X1)
        X1_sm = self.to_sparse(X1_emb)
        
        if X2 is None:
            X2 = X1
        X2_emb = self.neighbour_embed_data(X2)
        X2_sm = self.to_sparse(X2_emb)

        # Reshape matrices if the sizes are different
        nadd_row = abs(X1_sm.shape[0] - X2_sm.shape[0])
        if X1_sm.shape[0] > X2_sm.shape[0]:
            add_row = sparse.coo_matrix(([0], ([nadd_row-1], [X2_sm.shape[1]-1])))
            X2_sm = sparse.vstack((X2_sm, add_row))
        elif X1_sm.shape[0] < X2_sm.shape[0]:
            add_row = sparse.coo_matrix(([0], ([nadd_row - 1], [X1_sm.shape[1] - 1])))
            X1_sm = sparse.vstack((X1_sm, add_row))

        G = (X1_sm.T * X2_sm).todense().astype('float')
        
        if self.normalize:
            G /= np.array(np.sqrt(X1_sm.power(2).sum(0)))[0,:,None]
            G /= np.array(np.sqrt(X2_sm.power(2).sum(0)))[0,None,:]
            
        return G


# 4) Here is the all functions to compute the kmer and the neighbors function that we use in Mismatch kernel

def create_kmer_set(X, k, kmer_set={}):
    """
    Return a set of all kmers appearing in the dataset.
    """
    len_seq = len(X[0])
    idx = len(kmer_set)
    for i in range(len(X)):
        x = X[i]
        kmer_x = [x[i:i + k] for i in range(len_seq - k + 1)]
        for kmer in kmer_x:
            if kmer not in kmer_set:
                kmer_set[kmer] = idx
                idx += 1
    return kmer_set


def m_neighbours(kmer, m, recurs=0):
    """
    Return a list of neighbours kmers (up to m mismatches).
    """
    if m == 0:
        return [kmer]

    letters = ['G', 'T', 'A', 'C']
    k = len(kmer)
    neighbours = m_neighbours(kmer, m - 1, recurs + 1)

    for j in range(len(neighbours)):
        neighbour = neighbours[j]
        for i in range(recurs, k - m + 1):
            for l in letters:
                neighbours.append(neighbour[:i] + l + neighbour[i + 1:])
    return list(set(neighbours))


def get_neighbours(kmer_set, m):
    """
    Find the neighbours given a set of kmers.
    """
    kmers_list = list(kmer_set.keys())
    kmers = np.array(list(map(list, kmers_list)))
    num_kmers, kmax = kmers.shape
    neighbours = {}
    for i in range(num_kmers):
        neighbours[kmers_list[i]] = []

    for i in tqdm(range(num_kmers)):
        kmer = kmers_list[i]
        kmer_neighbours = m_neighbours(kmer, m)
        for neighbour in kmer_neighbours:
            if neighbour in kmer_set:
                neighbours[kmer].append(neighbour)
    return neighbours


def load_neighbors(dataset, k, m):
    """
    dataset: 0, 1 or 2\\
    k: len of the kmers
    m: number of possible mismatches
    """
    directory = 'saved1_neighbors/'
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    file_name = 'neighbours_'+str(dataset)+'_'+str(k)+'_'+str(m)+'.p'
    
                  
    # Load
    neighbours, kmer_set = pickle.load(open('saved_neighbors/'+file_name, 'rb'))
    print('Neighbors correctly loaded!')
    return neighbours, kmer_set


def load_or_compute_neighbors(dataset,k,m):
    """
    dataset: 0, 1 or 2\\
    k: len of the kmers
    m: number of possible mismatches
    """
    
    try:
        #Load the neighbors
        neighbours, kmer_set = load_neighbors(dataset, k, m)
    except:
        print('No file found, creating kmers neighbors')
        #Compute the neighbors
        directory = 'saved_neighbors/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_name = 'neighbours_'+str(dataset)+'_'+str(k)+'_'+str(m)+'.p'
        if dataset==0:
            X0_train = pd.read_csv("/Users/godchrist/Downloads/kernel-methods-ammi-2023/Xtr0.csv", sep=",", index_col=0).values
            X0_test =  pd.read_csv("/Users/godchrist/Downloads/kernel-methods-ammi-2023/Xte0.csv", sep=",", index_col=0).values
            kmer_set = create_kmer_set(X0_train[:,0], k, kmer_set={})
            kmer_set = create_kmer_set(X0_test[:,0], k, kmer_set)
            neighbours = get_neighbours(kmer_set, m)
            pickle.dump([neighbours, kmer_set], open('saved_neighbors/'+file_name, 'wb'))
        elif dataset==1:
            X1_train = pd.read_csv("/Users/godchrist/Downloads/kernel-methods-ammi-2023/Xtr1.csv", sep=",", index_col=0).values
            X1_test =  pd.read_csv("/Users/godchrist/Downloads/kernel-methods-ammi-2023/Xte1.csv", sep=",", index_col=0).values
            kmer_set = create_kmer_set(X1_train[:,0], k, kmer_set={})
            kmer_set = create_kmer_set(X1_test[:,0], k, kmer_set)
            neighbours = get_neighbours(kmer_set, m)
            pickle.dump([neighbours, kmer_set], open('saved_neighbors/'+file_name, 'wb'))
        elif dataset==2:
            X2_train = pd.read_csv("/Users/godchrist/Downloads/kernel-methods-ammi-2023/Xtr2.csv", sep=",", index_col=0).values
            X2_test = pd.read_csv("/Users/godchrist/Downloads/kernel-methods-ammi-2023/Xte2.csv", sep=",", index_col=0).values
            kmer_set = create_kmer_set(X2_train[:,0], k, kmer_set={})
            kmer_set = create_kmer_set(X2_test[:,0], k, kmer_set)
            neighbours = get_neighbours(kmer_set, m)
            pickle.dump([neighbours, kmer_set], open('saved_neighbors/'+file_name, 'wb'))
            
    return neighbours, kmer_set


# 5) Here are the parameters that we need to compute the 

C = 1.0 #Parameter C for SVM
k = 12 #Parameter k for SVM (only for 'spectrum' and 'mismatch')
m = 2 #Parameter m for SVM (only for 'mismatch')
list_k = [5,8,10,12,13,15] #List of parameters k for sum of mismatch kernels 
list_m = [1,1,1,2,2,3] #List of parameters m for sum of mismatch kernels 
weights = [1.0,1.0,1.0,1.0,1.0,1.0] #List of weights for sum of mismatch kernels

# 6) Let's apply now the Sum Mismatch kernel to all of the dataset

# a) Sum kernel on dataset 0
dataset_nbr = 0
kernels = []
for k,m in zip(list_k,list_m):
    neighbours, kmer_set = load_or_compute_neighbors(dataset_nbr, k, m)
    kernels.append(MismatchKernel(k=k, m=m, neighbours=neighbours, kmer_set=kmer_set, normalize = True))
svm = SVM(kernel=SumKernel(kernels=kernels, weights=weights), C=C)
svm.fit(X0_train, Y0_train)
pred0S = svm.predict_classes(X0_test)

# b) Sum kernel on dataset 1
dataset_nbr = 1
kernels = []
for k,m in zip(list_k,list_m):
    neighbours, kmer_set = load_or_compute_neighbors(dataset_nbr, k, m)
    kernels.append(MismatchKernel(k=k, m=m, neighbours=neighbours, kmer_set=kmer_set, normalize = True))
svm = SVM(kernel=SumKernel(kernels=kernels, weights=weights), C=C)
svm.fit(X1_train, Y1_train)
pred1S = svm.predict_classes(X1_test)

# c) Sum kernel on dataset 2
dataset_nbr = 2
kernels = []
for k,m in zip(list_k,list_m):
    neighbours, kmer_set = load_or_compute_neighbors(dataset_nbr, k, m)
    kernels.append(MismatchKernel(k=k, m=m, neighbours=neighbours, kmer_set=kmer_set, normalize = True))
svm = SVM(kernel=SumKernel(kernels=kernels, weights=weights), C=C)
svm.fit(X2_train, Y2_train)
pred2S = svm.predict_classes(X2_test)


# d) CREATE SUM MISMATCH KERNEL SUBMISSION FILE ####

pred = np.concatenate([pred0S.squeeze(),pred1S.squeeze(),pred2S.squeeze()])
pred = np.where(pred == -1, 0, 1)
pred_df = pd.DataFrame()
print(pred.shape)
print(pred)
pred_df['Bound'] = pred
pred_df.index.name = 'Id'

pred_df.to_csv('Yte.csv', sep=',', header=True)

#Public score of Sum Mismatch Kernel: 0.68266
#Private score of Sum Mismatch Kernel: 0.676