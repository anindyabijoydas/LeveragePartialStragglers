%   Finding the worst case condition number for beta-level matrix-vector scheme
%   We have n workers and s = c*ell - beta stragglers.
%   Storage fraction gamma = a1/a2.
%   Worst case condition number depends on the random matrix random_mat.
%   Different simulations may provide different worst case condition
%   numbers and different worst choice of workers.

clc
close all
clear

a1 = 1;
a2 = 10;
c = 3;
beta = 3;

gamma = a1/a2;
n = c*a2;
Delta = beta*a2;
ell = Delta*gamma;

s = c*ell - beta;
k = n - s;
random_mat = randn(n,beta);

CM = zeros(n,ell,beta);
for i=1:n
    for j=1:ell
        ee = rem((i+j-2)*beta+1:(i+j-1)*beta,Delta);
        ee(ee==0) = Delta;
        CM(i,j,:) = ee;
    end
end
for i = 1:n
    for j = 1:ell
        aa = zeros(Delta,1);
        aa(CM(i,j,:)) = random_mat(i,:);
        Coding_matrix{i}(j,:)= aa;
    end
end
                
choices = combnk(1:n,k);
condition_no = zeros(nchoosek(n,k),1);
for kk = 1:length(choices)
    wor = choices(kk,:);
    R = [];
    for i = 1:k
        R = [R ; Coding_matrix{wor(i)}];
    end
    condition_no(kk) = cond(R); 
end
worst_condition_no = max(condition_no);
pos = find(condition_no == worst_condition_no);

M1 = ['The recovery threshold is ', num2str(k),'.'];
fprintf('\n'); disp(M1);
M2 = ['The worst case condition number is ', num2str(worst_condition_no),'.'];
fprintf('\n'); disp(M2);
M3 = ['The worst case includes workers ', num2str(choices(pos,:)),'.'];
fprintf('\n'); disp(M3);
fprintf('\n');
