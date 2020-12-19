%   Finding the worst case condition number for optimal matrix-matrix scheme
%   We have n workers and s = n - kA * kB stragglers.
%   Storage fraction gammaA = 1/kA and gammaB = 1/kB.
%   ellu is the number of uncoded blocks in each worker.
%   ellc is the number of coded blocks in each worker.
%   Worst condition number depends on the random matrices random_matA and random_matB.
%   Different simulations may provide different worst case condition
%   numbers and different worst choice of workers.
%   One can increase the number of trials to find a smaller worst case 
%   condition number.

clc
close all
clear

n = 24;                                             % Number of workers
kA = 4;
kB = 5;
gammaA = 1/kA;
gammaB = 1/kB;
DeltaA = lcm(n,1/gammaA);
DeltaB = kB;
ellAu = DeltaA*DeltaB/n;
ellc = DeltaA*gammaA - ellAu;
k = kA*kB;
choices = combnk(1:n,k);
condition_no = zeros(nchoosek(n,k),1);
no_trials = 10;                                      % Number of trials

for trial = 1:no_trials
    random_matA{trial} = randn(n*ellc,DeltaA);      % Random matrix for A
    ind = 0;
    for i = 1:n
        ee = zeros(1,DeltaA);
        ee((i-1)*DeltaA/n+1) = 1;
        for j = 1:ellAu
            Coding_matrixA{i}(j,:)= ee;
            ee = circshift(ee,1);
        end
        for j = 1:ellc
            ind = ind + 1;
            Coding_matrixA{i}(j+ellAu,:)= random_matA{trial}(ind,:);
        end
    end
    
    random_matB{trial} = randn(n,DeltaB);           % Random matrix for B
    for i = 1:n
        Coding_matrixB{i} = random_matB{trial}(i,:);
    end
    
    for i =1:n
        Coding_matrixAB{i} = kron(Coding_matrixA{i},Coding_matrixB{i});
    end
    
    for kk = 1:length(choices)
        wor = choices(kk,:);
        R = [];
        for i = 1:k
            R = [R ; Coding_matrixAB{wor(i)}];
        end
        condition_no(kk) = cond(R);
    end
    worst_condition_no(trial) = max(condition_no);
    pos = condition_no == worst_condition_no(trial);
    worst_choice_of_workers(trial,:) = choices(pos,:);
end
worst_cond_no_over_trials = min(worst_condition_no);
pos = find(worst_condition_no == worst_cond_no_over_trials);

M1 = ['The worst case condition number is ', num2str(worst_cond_no_over_trials),'.'];
fprintf('\n'); disp(M1);
M2 = ['The worst case includes workers ', num2str(worst_choice_of_workers(pos,:)),'.'];
fprintf('\n'); disp(M2);
fprintf('\n');
