%   Finding the worst case condition number for optimal matrix-vector scheme
%   We have n workers and s = n - k stragglers.
%   Storage fraction gamma = 1/k.
%   ellu is the number of uncoded blocks in each worker.
%   ellc is the number of coded blocks in each worker.
%   Worst case condition number depends on the random matrix random_mat.
%   Different simulations may provide different worst case condition
%   numbers and different worst choice of workers.
%   One can increase the number of trials to find a smaller worst case 
%   condition number.

clc
close all
clear

n = 18;                                      % The number of workers
k = 15;      
gamma = 1/k;                                 % Storage fraction
Delta = lcm(n,1/gamma);
ellu = Delta/n;                              % Number of uncoded blocks
ellc = Delta*gamma - ellu;                   % Number of coded blocks
no_trials = 10;                               % Number of trials
choices = combnk(1:n,k);
condition_no = zeros(nchoosek(n,k),1);
worst_choice_of_workers = zeros(no_trials,k);

for trial = 1 : no_trials
    random_mat{trial} = randn(n*ellc,Delta);           % Random Matrix
    ind = 0;
    for i = 1:n
        ee = zeros(1,Delta);
        ee((i-1)*Delta/n+1) = 1;
        for j = 1:ellu
            Coding_matrix{i}(j,:)= ee;
            ee = circshift(ee,1);
        end
        for j = 1:ellc
            ind = ind + 1;
            Coding_matrix{i}(j+ellu,:)= random_mat{trial}(ind,:);
        end
    end
    
    for kk = 1:length(choices)
        wor = choices(kk,:);
        R = [];
        for i = 1:k
            R = [R ; Coding_matrix{wor(i)}];
        end
        condition_no(kk) = cond(R);
        rank_R(kk) = rank(R);
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
