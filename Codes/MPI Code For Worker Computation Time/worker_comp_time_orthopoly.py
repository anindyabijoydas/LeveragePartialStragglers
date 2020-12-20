"""
Finding the worker computation time for the Ortho-Poly Codes
Having two random matrices A and B
We have n workers and s = n - kA * kB stragglers.
Storage fraction gammaA = 1/kA and gammaB = 1/kB.
Matrices A and B are partitioned into kA and kB block columns.

This code uses the approach of the following paper-

Fahim, Mohammad, and Viveck R. Cadambe. "Numerically stable polynomially coded 
computing." In 2019 IEEE International Symposium on Information Theory (ISIT), 
pp. 3017-3021. IEEE, 2019.
"""


from __future__ import division
import numpy as np
import itertools as it
import time
from scipy.sparse import csr_matrix
from scipy.sparse import rand,vstack
from mpi4py import MPI
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    n = 24 ;                                       # Number of workers
    kA = 4;
    kB = 5;
    k = kA*kB;
    s = n - k                                      # Number of stragglers
    r = 15000 ;
    t = 12000 ;
    w = 13500 ;

    A = rand(t, r, density=0.02, format="csr")     # sparse A
    B = rand(t, w, density=0.02, format="csr")     # sparse B
    A = A.todense()
    B = B.todense()
    E = np.matmul(np.transpose(A),B);
    
    rho = np.zeros((n,1),dtype=float)
    TA = np.zeros((n,kA),dtype=float) 
    TB = np.zeros((n,kB),dtype=float)

    for i in range(0,n):
        rho[i] = np.cos((2*i+1)*np.pi/(2*n))
    
    for rr in range(0,kA):
        dd = np.cos(rr*np.arccos(rho))
        TA[:,rr] = dd[:,0]

    for rr in range(0,kB):
        dd = np.cos(rr*kA*np.arccos(rho))
        TB[:,rr] = dd[:,0]

    TA[:,0] = TA[:,0]/np.sqrt(2);
    TB[:,0] = TB[:,0]/np.sqrt(2);    

    T = np.zeros((n,kA*kB),dtype=float)
    for i in range(0,n):
        T[i,:] = np.kron(TA[i,:],TB[i,:])

    workers = list(range(n));         
    Coding_A = TA;    
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

    Coding_B = TB;    
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
    sending_time = np.zeros(n,dtype = float); 
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
        
