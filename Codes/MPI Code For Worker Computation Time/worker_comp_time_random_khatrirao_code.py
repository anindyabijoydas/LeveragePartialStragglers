"""
Finding the worker computation time for random khatri-rao code approach.
Having two random matrices A and B
We have n workers and s = n - kA * kB stragglers.
Storage fraction gammaA = 1/kA and gammaB = 1/kB.
Matrices A and B are divided into kA and kB block columns.
One can change the number of trials to find better coefficients.


This code uses the approach of the following paper-

Subramaniam, Adarsh M., Anoosheh Heidarzadeh, and Krishna R. Narayanan. 
"Random Khatri-Rao-Product Codes for Numerically-Stable Distributed Matrix Multiplication." 
In 2019 57th Annual Allerton Conference on Communication, Control, and Computing (Allerton), 
pp. 253-259. IEEE, 2019.
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

def mat_mat_best_rand(n,kA,kB,no_trials):
    condition_no = np.zeros(no_trials,dtype=float);
    Rr_A = {};
    Rr_B = {};
    RintA = np.zeros((k,kA),dtype = float);
    for i in range (0,kB):
        RintA[i*kA:(i+1)*kA,:] = np.identity(kA,dtype = float);
    RintB = np.zeros((k,kB),dtype = float);
    for i in range (0,kB):
        RintB[i*kA:(i+1)*kA,i] = 1;
    s = n - kA*kB;       
    print('\n')
    for mm in range(0,no_trials):
        print('Trial %s is running' %(mm+1))
        Rr_A[mm] = np.random.normal(0, 1, [s,kA]);
        R_A = np.concatenate((RintA,Rr_A[mm]), axis = 0)
        Rr_B[mm] = np.random.normal(0, 1, [s,kB]);
        R_B = np.concatenate((RintB,Rr_B[mm]), axis = 0)

        R_AB = np.zeros((n,k),dtype = float);
        for i in range(0,n):
            R_AB[i,:] = np.kron(R_A[i,:],R_B[i,:]);

        workers = np.array(list(range(n)));
        Choice_of_workers = list(it.combinations(workers,k));
        size_total = np.shape(Choice_of_workers);            
        total_no_choices = size_total[0];
        cond_no = np.zeros(total_no_choices,dtype = float);    

        for i in range (0, total_no_choices):
            dd = list(Choice_of_workers[i]); 
            Coding_matrix = R_AB[dd,:]
            cond_no[i] = np.linalg.cond(Coding_matrix);
        condition_no[mm] = np.max(cond_no);
    pos =   np.argmin(condition_no); 
    R_A = np.concatenate((RintA,Rr_A[pos]), axis = 0)
    R_B = np.concatenate((RintB,Rr_B[pos]), axis = 0)
    best_cond_min = condition_no[pos];
    return R_A,R_B,best_cond_min

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

    RintA = np.zeros((k,kA),dtype = float);
    for i in range (0,kB):
        RintA[i*kA:(i+1)*kA,:] = np.identity(kA,dtype = float);
    RintB = np.zeros((k,kB),dtype = float);
    for i in range (0,kB):
        RintB[i*kA:(i+1)*kA,i] = 1;
    R_A = np.concatenate((RintA,np.random.normal(0, 1, [s,kA])), axis = 0)
    R_B = np.concatenate((RintB,np.random.normal(0, 1, [s,kB])), axis = 0)

    #no_trials = 0;              # One can comment out these two lines to find good choices of coefficients
    #(R_A,R_B,best_cond_min) = mat_mat_best_rand(n,kA,kB,no_trials);
    
    R_AB = np.zeros((n,k),dtype = float);
    for i in range(0,n):
        R_AB[i,:] = np.kron(R_A[i,:],R_B[i,:]);

    workers = np.array(list(range(n)));
    Choice_of_workers = list(it.combinations(workers,k));
    size_total = np.shape(Choice_of_workers);            
    total_no_choices = size_total[0];
    cond_no = np.zeros(total_no_choices,dtype = float);    

    for i in range (0, total_no_choices):
        dd = list(Choice_of_workers[i]); 
        Coding_matrix = R_AB[dd,:]
        cond_no[i] = np.linalg.cond(Coding_matrix);
  
    worst_condition_number = np.max(cond_no);
    pos =   np.argmax(cond_no);
    worst_choice_of_workers = list(Choice_of_workers[pos]);
    print('Worst condition Number is %s' % worst_condition_number)
    print('Worst Choice of workers set includes workers %s ' % worst_choice_of_workers)

    Coding_A = R_A;    
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

    Coding_B = R_B;    
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

