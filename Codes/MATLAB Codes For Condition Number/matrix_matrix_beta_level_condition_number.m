%   Finding the worst case condition number for beta-level matrix-matrix scheme
%   We have n workers and s = c*ell - beta stragglers.
%   ell = ellA * ellB is the number of assigned blocks in each worker.
%   Worst condition number depends on the random matrices random_matA and random_matB.
%   Different simulations may provide different worst case condition
%   numbers and different worst choice of workers.

clc
close all
clear

a1 = 1;
a2 = 3;
b1 = 1;
b2 = 3;
c = 2;

gammaA = a1/a2;
gammaB = b1/b2;
n = c*a2*b2;                                % Number of workers
betaA = 2;
betaB = 2;

DeltaA = betaA*a2;
ellA = DeltaA*gammaA;
DeltaB = betaB*b2;
ellB = DeltaB*gammaB;
ell = ellA*ellB;
beta = betaA*betaB;
s = c*ell - beta;                           % Number of stragglers
k = n - s;

random_matA = randn(n,betaA);
CMA = zeros(n,ellA,betaA);
for i=1:n
    for j=1:ellA
        ee = rem((i+j-2)*betaA+1:(i+j-1)*betaA,DeltaA);
        ee(ee==0) = DeltaA;
        CMA(i,j,:) = ee;
    end
end

for i = 1:n
    for j = 1:ellA
        aa = zeros(DeltaA,1);
        aa(CMA(i,j,:)) = random_matA(i,:);
        Coding_matrixA{i}(j,:)= aa;
    end
end

random_matB = randn(n,betaB);
CMB = zeros(n,ellB,betaB);
for i=1:n
    ii = floor((i-1)*betaA/DeltaA) + 1;
    ii = rem(ii,DeltaB);
    if ii == 0
        ii = DeltaB;
    end
    
    for j=1:ellB
        ee = rem((ii+j-2)*betaB+1:(ii+j-1)*betaB,DeltaB);
        ee(ee==0) = DeltaB;
        CMB(i,j,:) = ee;
    end
end

for i = 1:n
    for j = 1:ellB
        aa = zeros(DeltaB,1);
        aa(CMB(i,j,:)) = random_matB(i,:);
        Coding_matrixB{i}(j,:)= aa;
    end
end

for i =1:n
    Coding_matrixAB{i} = kron(Coding_matrixA{i},Coding_matrixB{i});
end

choices = combnk(1:n,k);
condition_no = zeros(nchoosek(n,k),1);
for kk = 1:length(choices)
    wor = choices(kk,:);
    R = [];
    for i = 1:k
        R = [R ; Coding_matrixAB{wor(i)}];
    end
    condition_no(kk) = cond(R);   
end
worst_condition_no = max(condition_no);
pos = find(condition_no == worst_condition_no);

M1 = ['The worst case condition number is ', num2str(worst_condition_no),'.'];
fprintf('\n'); disp(M1);
M2 = ['The worst case includes workers ', num2str(choices(pos,:)),'.'];
fprintf('\n'); disp(M2);
fprintf('\n');
