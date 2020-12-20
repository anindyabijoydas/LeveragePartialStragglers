%   Finding the worst case condition number for coded at bottom matrix-vector scheme
%   We have n workers and s stragglers.
%   Storage fraction gamma = a1/a2.
%   ellu is the number of uncoded blocks in each worker.
%   ellc is the number of coded blocks in each worker.
%   Worst case condition number depends on the random matrix random_mat.
%   Different simulations may provide different worst case condition
%   numbers and different worst choice of workers.

clc
close all
clear

au = 2;
ac = 1;
a1 = au + ac;
a2 = 5;

gammau = au/a2;
gammac = ac/a2;
m = 1;
n = m * a2;                                 % Number of workers
DeltaA = n;
s = floor((n^2*gammac + n*gammau - 1)/(n*gammac + 1));
k = n - s;                                  % recovery threshold

ellu = DeltaA*gammau;                       % Number of uncoded blocks
ellc = DeltaA*gammac;                       % Number of coded blocks
ell = ellu + ellc;

zellu = zeros(ellu,DeltaA);
zellu(1:ellu,1:ellu) = eye(ellu);
random_mat = randn(n*ellc,DeltaA);
for i = 1:n
    Wa{i}(1:ellu,:) = zellu;
    zellu = (circshift(zellu',1))';
    Wa{i}(ellu+1:ellu+ellc,:) = random_mat((i-1)*ellc+1:i*ellc,:);
end

workers = 1:n;
Choice_of_workers = combnk(workers,k);
[total_no_choices,~] = size(Choice_of_workers);
cond_no = zeros(total_no_choices,1);

for i = 1:total_no_choices
    Coding_matrix = [];
    for j = 1:k
        Coding_matrix = [Coding_matrix ; Wa{Choice_of_workers(i,j)}];
    end
    cond_no(i) = cond(Coding_matrix);
end

worst_cond_no = max(cond_no);
pos = find(cond_no == worst_cond_no);

M1 = ['The worst case condition number is ', num2str(worst_cond_no),'.'];
fprintf('\n'); disp(M1);
M2 = ['The worst case includes workers ', num2str(Choice_of_workers(pos,:)),'.'];
fprintf('\n'); disp(M2);
fprintf('\n');