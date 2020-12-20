%   Finding the worst case condition number for uncoded matrix-vector scheme
%   We have n workers and s = c*ell - 1 stragglers.
%   Storage fraction gamma = a1/a2.

clc
close all
clear

a1 = 1;
a2 = 10;
c = 3;
beta = 1;                                   % uncoded indicates beta = 1

gamma = a1/a2;                              % storage fraction
n = c*a2;
Delta = beta*a2;
ell = Delta*gamma;

s = c*ell - beta;
k = n - s;

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
        aa(CM(i,j,:)) = 1;
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
M3 = ['The worst case includes workers ', num2str(choices(pos(1),:)),'.'];
fprintf('\n'); disp(M3);
fprintf('\n');
