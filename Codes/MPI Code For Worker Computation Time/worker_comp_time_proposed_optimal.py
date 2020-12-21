"""
Finding the worker computation time for the proposed optimal scheme
Having two random matrices A and B
We have n workers and s = n - kA * kB stragglers.
Storage fraction gammaA = 1/kA and gammaB = 1/kB.
Matrices A and B are partitioned into kA and kB block columns.
ellAu and ellAc are number of uncoded and coded blocks
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
    n = 24;
    gammaA = 1/4;
    gammaB = 1/5;
    kA = int(1/gammaA)
    kB = int(1/gammaB)
    
    r = 15000 ;
    t = 12000 ;
    w = 13500 ;
    A = rand(r, t, density=0.02, format="csr")
    B = rand(w, t, density=0.02, format="csr")
    DeltaA = 24
    ellA = int(DeltaA*gammaA);
    DeltaB = kB
    ellB = 1;
    ellAu = int(DeltaA/n*kB);
    ellAc = ellA - ellAu  
    
    sub_ellA = int(r/DeltaA);
    As = {};
    for j in range(0,DeltaA):
        As[j] = A[j*sub_ellA:(j+1)*sub_ellA,:];
    
    sub_ellB = int(w/DeltaB);
    Bs = {};
    for j in range(0,DeltaB):
        Bs[j] = B[j*sub_ellB:(j+1)*sub_ellB,:];
    
    CMA = np.zeros((n,ellAu))
    for i in range(0,n):
        kk = int(i*DeltaA/n)
        for j in range(0,ellAu):
            CMA[i,j] = np.mod(np.arange((kk+j),(kk+j+1)),DeltaA)
       
    res_finAs = {}
    res_finAd = {}        
    res_finB = {}        
    Av = {};
    for j in range(0,DeltaA):
        Av[j] = csr_matrix.reshape(As[j],(1,sub_ellA*t))
    Ac = Av[0];    
    for j in range(1,DeltaA):
        Ac = vstack([Ac,Av[j]]) 
    
    Bv = {};
    for j in range(0,DeltaB):
        Bv[j] = csr_matrix.reshape(Bs[j],(1,sub_ellB*t))
    Bc = Bv[0];    
    for j in range(1,DeltaB):
        Bc = vstack([Bc,Bv[j]])  
        
    for k in range(0,n):
        resA = {};
        resB = {};
        for i in range(0, ellAu):
            resA[i] = As[CMA[k,i]].todense()
        for i in range(0, ellAc):
            vecA = np.random.rand(1,DeltaA)
            coded_submatA = vecA * Ac
            resA[i+ellAu] = coded_submatA.reshape((sub_ellA, t))
        vecB = np.random.rand(1,DeltaB)
        coded_submatB = vecB * Bc
        resB = coded_submatB.reshape((sub_ellB, t))
        
        res_finAs[k] = np.vstack((resA[i]) for i in range (0,ellAu))
        res_finAs[k] = csr_matrix(res_finAs[k])
        res_finAd[k] = np.vstack((resA[i]) for i in range (ellAu,ellA))
        res_finB[k] = csr_matrix(resB)
        res_finB[k] = csr_matrix.transpose(res_finB[k])
        
    
    for k in range (0,n):
        comm.send(res_finAs[k], dest=k+1)
        comm.send(res_finAd[k], dest=k+1)
        comm.send(res_finB[k], dest=k+1)

    computation_time = np.zeros(n,dtype = float); 
    for i in range (0,n):
        computation_time[i] = comm.recv(source=i+1);

    for i in range (0,n):
        print("Computation time (in optimal scheme) for processor %s is %s" %(i,computation_time[i]))

    comm.Abort()

else:
    smatAs = comm.recv(source=0);
    smatAd = comm.recv(source=0);
    smatB = comm.recv(source=0);
    smatBd = smatB.todense()

    start_time = time.time()
    result = np.concatenate(((smatAs * smatB).todense(),smatAd * smatBd),axis=0)
    end_time = time.time()
    comm.send(end_time - start_time, dest=0)

