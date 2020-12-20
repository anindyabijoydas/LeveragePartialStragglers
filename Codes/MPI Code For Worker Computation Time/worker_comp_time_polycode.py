"""
Finding the worker computation time for polynomial code approach.
Having a random matrices A and B
We have n workers, s stragglers.
Storage fraction gamma = 1/k ; where k = n - s; and set Delta = k.
Matrix A is divided into k block columns.
We choose n nodes uniformly spaced in [-1,1], instead of the integers.
One can find MSE at different SNR

This code uses the approach of the following paper-

Qian Yu, Mohammad Maddah-Ali, and Salman Avestimehr. Polynomial codes: an 
optimal design for highdimensional coded matrix multiplication. In Proc. of 
Advances in Neural Information Processing Systems (NIPS), pages 4403–4413, 2017
"""

from __future__ import division
import numpy as np
import itertools as it
import time
from mpi4py import MPI
from scipy.sparse import csr_matrix
from scipy.sparse import rand,vstack
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    n = 24 ;                                        # Number of worker nodes
    kA = 4;
    kB = 5;
    k = kA*kB;
    s = n - k ;                                     # Number of stragglers
    r = 15000 ;
    t = 12000 ;
    w = 13500 ;
    A = rand(t, r, density=0.02, format="csr")	    # sparse A
    B = rand(t, w, density=0.02, format="csr")	    # sparse B
    A = A.todense()
    B = B.todense()
    E = np.matmul(np.transpose(A),B);
    
    node_points = -1+2*(np.array(list(range(n)))+1)/n;
    nodes = node_points[np.arange(k)];
    Coding_matrix = np.zeros((k,k),dtype = float);
    Coding_matrix1 = np.zeros((n,k),dtype = float);
    
    for j in range (0,k):
        Coding_matrix[j,:] = (nodes[j])**np.array(list(range(k)));
    for j in range (0,n):
        Coding_matrix1[j,:] = (node_points[j])**np.array(list(range(k)));
        
    Coding_A = Coding_matrix1[:,::kB];    
    c = int(r/kA);
    W1a = {};
    for i in range (0,kA):
        W1a[i] = A[:,i*c:(i+1)*c];

    W2a = {} ;
    (uu,vv) = np.shape(W1a[0]);
    for i in range (0,n):
        W2a[i] = np.zeros((uu,vv),dtype=float); 
        for j in range (0,kA):
            W2a[i] = W2a[i] + Coding_A[i,j]*W1a[j];

    Coding_B = Coding_matrix1[:,0:kB];    
    d = int(w/kB);
    W1b = {};
    for i in range (0,kB):
        W1b[i] = B[:,i*d:(i+1)*d];

    W2b = {} ;
    (uu,vv) = np.shape(W1b[0]);
    for i in range (0,n):
        W2b[i] = np.zeros((uu,vv),dtype=float); 
        for j in range (0,kB):
            W2b[i] = W2b[i] + Coding_B[i,j]*W1b[j];
    work_product = {};
    
    for i in range (0,n):
        Ai = W2a[i];
        Bi = W2b[i];
        comm.send(Ai, dest=i+1)
        comm.send(Bi, dest=i+1)
    
    computation_time_dense = np.zeros(n,dtype = float); 
    computation_time_sparse = np.zeros(n,dtype = float); 

    for i in range (0,n):
        computation_time_dense[i] = comm.recv(source=i+1);
        computation_time_sparse[i] = comm.recv(source=i+1);

    for i in range (0,n):
        print("Computation time (using sparse method) for processor %s is %s" %(i,computation_time_sparse[i]))

    print('\n')

    for i in range (0,n):
        print("Computation time (using regular method) for processor %s is %s" %(i,computation_time_dense[i]))

    comm.Abort()

else:
    Ai = comm.recv(source=0)
    Bi = comm.recv(source=0)
    Ait = np.transpose(Ai)

    start_time = time.time()
    Wab = Ait * Bi;						# Ait and Bi are dense 
    end_time = time.time();
    comp_time_dense = end_time - start_time;
    comm.send(comp_time_dense, dest=0)

    Ait = csr_matrix(Ait)							
    Bi = csr_matrix(Bi)
    start_time = time.time()
    Wab = Ait * Bi;						# Ait and Bi are sparse
    end_time = time.time();
    comp_time_sparse = end_time - start_time;
    comm.send(comp_time_sparse, dest=0) 
