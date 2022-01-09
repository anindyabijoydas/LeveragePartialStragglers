"""
Finding the worker computation time for the proposed beta-level coding scheme
Having two random matrices A and B
We have n workers and s stragglers.
Storage fraction gammaA = a1/a2 and gammaB = b1/b2.
Matrices A and B are partitioned into a2*betaA and b2*betaB block columns.
ellA and ellB are the number of coded blocks for A and B, respectively.
"""

from __future__ import division
import numpy as np
import time
from scipy.sparse import csr_matrix
from mpi4py import MPI
import sys
import warnings
from scipy.sparse import rand,vstack

if not sys.warnoptions:
    warnings.simplefilter("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    n = 18;
    a1 = 1;
    a2 = 3;
    b1 = 1;
    b2 = 3;
    gammaA = a1/a2;
    gammaB = b1/b2;
    r = 13680;
    t = 12000;
    w = 10260;
    A = rand(r, t, density=0.05, format="csr")
    B = rand(w, t, density=0.05, format="csr")
    betaA = 2;
    betaB = 2;
    DeltaA = int(betaA*a2)
    DeltaB = int(betaB*b2)
    ellA = int(DeltaA*gammaA);
    ellB = int(DeltaB*gammaB);


    sub_ellA = int(r/DeltaA);
    As = {};
    for j in range(0,DeltaA):
        As[j] = A[j*sub_ellA:(j+1)*sub_ellA,:];

    sub_ellB = int(w/DeltaB);
    Bs = {};
    for j in range(0,DeltaB):
        Bs[j] = B[j*sub_ellB:(j+1)*sub_ellB,:];
    
    CMA = np.zeros((n,ellA,betaA))
    for i in range(0,n):
        for j in range(0,ellA):
            CMA[i,j,:] = np.mod(np.arange(((i+j)*betaA),((i+j+1)*betaA)),DeltaA)

    CMB = np.zeros((n,ellB,betaB))
    for i in range(0,n):
        alphaA = DeltaA/betaA;
        k = np.floor(i/alphaA);
        for j in range(0,ellB):
            CMB[i,j,:] = np.mod(np.arange(((k+j)*betaB),((k+j+1)*betaB)),DeltaB)
    
    res_finA = {}        
    res_finB = {}        

    for k in range(0,n):
        resA = {};
        resB = {};
        vecA = np.random.rand(1,betaA)
        vecB = np.random.rand(1,betaB)  
        for i in range(0, ellA):
            Av = {};
            for j in range(0,betaA):
                Av[j] = csr_matrix.reshape(As[CMA[k,i,j]],(1,sub_ellA*t))
            Ac = Av[0];    
            for j in range(1,betaA):
                Ac = vstack([Ac,Av[i]])      
            coded_submatA = vecA * Ac
            resA[i] = coded_submatA.reshape((sub_ellA, t))
    
            Bv = {};
            for j in range(0,betaB):
                Bv[j] = csr_matrix.reshape(Bs[CMB[k,i,j]],(1,sub_ellB*t))
            Bc = Bv[0];    
            for j in range(1,betaB):
                Bc = vstack([Bc,Bv[i]])      
            coded_submatB = vecB * Bc
            resB[i] = coded_submatB.reshape((sub_ellB, t))
    
        res_finA[k] = np.vstack((resA[i]) for i in range (0,ellA))
        res_finA[k] = csr_matrix(res_finA[k])
        res_finB[k] = np.vstack((resB[i]) for i in range (0,ellB))
        res_finB[k] = csr_matrix(res_finB[k])
        res_finB[k] = csr_matrix.transpose(res_finB[k])
    
    for k in range (0,n):
        comm.send(res_finA[k], dest=k+1)
        comm.send(res_finB[k], dest=k+1)
    
    computation_time = np.zeros(n,dtype = float); 
    for i in range (0,n):
        computation_time[i] = comm.recv(source=i+1);

    for i in range (0,n):
        print("Computation time (using beta-level coding) for processor %s is %s" %(i,computation_time[i]))

    comm.Abort()  

else:        
    Ai = comm.recv(source=0)
    Bi = comm.recv(source=0)

    start_time = time.time()
    Wab = Ai * Bi;						
    end_time = time.time();
    comp_time_dense = end_time - start_time;
    comm.send(comp_time_dense, dest=0)

