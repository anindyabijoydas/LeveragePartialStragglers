from __future__ import division
import numpy as np
import itertools as it
import time
from scipy.sparse import csr_matrix
from scipy.sparse import rand,vstack,hstack
from scipy.linalg import eigvals
from mpi4py import MPI
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def shiftrow(A,r,t):   
    (a,b) = np.shape(A) ;
    B = np.zeros((t,b),dtype = float) ;
    B[r:a+r,:] = A ;
    return B;
def matrix_matrix_best_mat(n,kA,kB,gammaB,no_trials):
    samples_in_omega = 200;
    k = kA*kB;
    s = n - k;
    identity_part = np.identity((k),dtype = float);
    DeltaB = int(np.round((s-1)*(kB-1)/(gammaB-1/kB))); 
    while DeltaB % kB != 0:
        DeltaB = DeltaB+1 ;
    z = int(DeltaB/kB + (s-1)*(kB-1));
    workers = list(range(n));
    Choice_of_workers = list(it.combinations(workers,k));
    size_total = np.shape(Choice_of_workers);            
    total_no_choices = size_total[0];
    best_mat_A = {};
    best_mat_B = {};
    mu = 0;
    sigma = 1;
    min_eigenvalue = np.zeros(total_no_choices*samples_in_omega);
    max_eigenvalue = np.zeros(total_no_choices*samples_in_omega);
    condition_number = np.zeros(no_trials,dtype = float);
    for trial in range (0,no_trials):
        best_mat_A[trial] = np.random.normal(mu, sigma, [kA,s]);
        best_mat_B[trial] = np.random.normal(mu, sigma, [kB,s]);
        matrices = {};
        matrices[0]=best_mat_A[trial];
        matrices[1]=best_mat_B[trial];
        best_mat_comb = np.zeros((kA*kB, s))
        for i in range(s):
            cum_prod = matrices[0][:, i]
            cum_prod = np.einsum('i,j->ij', cum_prod, matrices[1][:, i]).ravel()
            best_mat_comb[:, i] = cum_prod  
        ind = 0;
        exponent_vector = list(range(0,kB)) ;
        for i in range (1,kA):
            m = i*z;
            exponent_vector = np.concatenate((exponent_vector,list(range(m,m+kB))),axis=0);
        w = np.zeros(2*samples_in_omega,dtype = float);
        for i in range (0,samples_in_omega):
            w[i] = -np.pi + i*2*np.pi/samples_in_omega;
        zz = samples_in_omega;
        for z in range (0,zz):
            imag = 1j;
            omega = np.zeros((k,s),dtype = complex);
            for i in range (0,s):
                omega[:,i] = np.power(np.exp(-imag*w[z])**i,list(range(k)))
            Generator_mat = np.concatenate((identity_part, np.multiply(best_mat_comb,omega)), axis = 1)
            for i in range (0,total_no_choices):
                Coding_matrix = [];
                kk = list(Choice_of_workers[i]);
                Coding_matrix = Generator_mat[:,kk];
                Coding_matrixT = np.transpose(Coding_matrix);
                D = np.matmul(np.conjugate(Coding_matrixT),Coding_matrix);
                eigenvalues = eigvals(D);
                eigenvalues = np.real(eigenvalues);
                min_eigenvalue[ind] = np.min(eigenvalues);
                max_eigenvalue[ind] = np.max(eigenvalues);
                ind = ind + 1;
        condition_number[trial] = np.sqrt(np.max(max_eigenvalue)/np.min(min_eigenvalue)) 
    best_cond_min = np.min(condition_number);
    position =   np.argmin(condition_number);
    R_A = best_mat_A[position];
    R_B = best_mat_B[position];
    return R_A,R_B,best_cond_min  

if rank == 0:
    n = 24;                                         # Number of workers
    kA = 4;
    kB = 5;
    k = kA*kB;
    s = n - k;                                      # Number of stragglers
    gammaA = 2/5 ;
    DeltaA = int(np.round((s-1)*(kA-1)/(gammaA-1/kA))); 
    while DeltaA % kA != 0:
        DeltaA = DeltaA+1 ;
    qA = int(DeltaA/kA) ;
    print("\nThe value of DeltaA is %s" %(DeltaA))
    gammaB = 1/3 ;
    DeltaB = int(np.round((s-1)*(kB-1)/(gammaB-1/kB))); 
    while DeltaB % kB != 0:
        DeltaB = DeltaB+1 ;
    qB = int(DeltaB/kB) ;
    print("The value of DeltaB is %s" %(DeltaB))
    r = 1500 ;
    t = 1200 ;
    w = 1350 ;

    A = rand(t, r, density=0.02, format="csr")
    B = rand(t, w, density=0.02, format="csr")
    A = A.todense()
    B = B.todense()
    E = np.matmul(np.transpose(A),B);
    worst_case = 0;
    R_A = np.random.normal(0,1, [kB,s]);  
    R_B = np.random.normal(0,1, [kB,s]);

    #no_trials = 0;              # One can comment out these two lines to find good choices of coefficients
    #(R_A,R_B,best_cond_min) = matrix_matrix_best_mat(n,kA,kB,gammaB,no_trials);
    
    if worst_case !=1:
        all_workers = np.random.permutation(n);
        active_workers = all_workers[0:k];
        active_workers.sort() ;
        print("\nActive Workers are %s"%(active_workers))

    aa = int(r/DeltaA);
    Wa = {};
    for i in range (0,DeltaA):
        Wa[i] = A[:,i*aa:(i+1)*aa];
    
    bb = int(w/DeltaB);
    Wb = {};
    for i in range (0,DeltaB):
        Wb[i] = B[:,i*bb:(i+1)*bb];

    Wa1 = {} ;
    for i in range (0,k):
        for j in range (0,qA):
            Wa1[i,j] = Wa[np.floor(i/kB)*qA+j];        

    for i in range (k,n):
        lenA = qA + (kA-1)*(i-k);
        for j in range (0,lenA):
            sumA = np.zeros((t,aa),dtype = float);
            for rr in range (0,kA):
                if j >= (i-k)*rr and j <= (i-k)*rr+qA-1:
                    sumA = sumA + Wa1[kB*rr,j-(i-k)*rr]*R_A[rr,i-k];                
            Wa1[i,j] = sumA;

    Wbb = {};
    for i in range (0,kB):
        for j in range (0,qB):
            Wbb[i,j] = Wb[i*qB+j];
    Wb1 = {};
    for i in range (0,k):
        for j in range (0,qB):
            Wb1[i,j] = Wbb[np.remainder(i,kB),j];

    for i in range (k,n):
        lenB = qB + (kB-1)*(i-k);
        for j in range (0,lenB):
            sumB = np.zeros((t,bb),dtype = float);
            for rr in range (0,kB):
                if j >= (i-k)*rr and j <= (i-k)*rr+qB-1:
                    sumB = sumB + Wbb[rr,j-(i-k)*rr]*R_B[rr,i-k];                
            Wb1[i,j] = sumB;
    Wab = {};
    WW = {};
    for i in range (0,n):
        lengA = qA if i < k else qA + (kA-1)*(i-k) ;
        WsendA = {};
        for j in range (0,lengA):
            WsendA[j] = Wa1[i,j];
        comm.send(WsendA, dest=i+1)
        comm.send(lengA, dest=i+1)
        lengB = qB if i < k else qB + (kB-1)*(i-k) ;
        WsendB = {};
        for j in range (0,lengB):
            WsendB[j] = Wb1[i,j];
        comm.send(WsendB, dest=i+1)
        comm.send(lengB, dest=i+1)

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
    WsendA = comm.recv(source=0)
    lengA = comm.recv(source=0)
    WsendB = comm.recv(source=0)
    lengB = comm.recv(source=0)

    Ai =  np.hstack(WsendA[j] for j in range(0,lengA))    
    Bi =  np.hstack(WsendB[j] for j in range(0,lengB))    
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
        


