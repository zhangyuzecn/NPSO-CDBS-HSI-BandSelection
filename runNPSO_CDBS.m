clear
%% Parameters for data
data_type_s1 = 'QY';  
data_type_s2 = 'PA';
data_type_t = 'TD';

% Load source and target datasets
[data1,label1,IE1] = load_data(data_type_s1);
[data2,label2,IE2] = load_data(data_type_s2);
[data3,label3,IE3] = load_data(data_type_t);
data{1} = data1;
data{2} = data2;
data{3} = data3;
label{1} = label1;
label{2} = label2;
label{3} = label3;

% Compute Euclidean distance between columns
distances = pdist(data3');
% Convert the distance vector into a similarity matrix
similarity_matrix = squareform(distances);

% Parameter settings
params.IE = IE3;
params.N_task = 2 ;
params.opt_type = 'PSO';    
params.N_bands = size(data3,2);    % Number of spectral bands
params.N_ind = 100;                % Number of individuals in the population
params.GX = 100;                   % Maximum number of iterations
params.Range = [5:5:50];
t_max = 20;  

% Pretraining parameters
params.N_mask = 200;               % Number of masks
hiddenSizes = [100];
resname = [num2str(t_max),' times results (hidden layer ', num2str(hiddenSizes), ')_', data_type_t, ...
            '(', data_type_s1, ',', data_type_s2, ')_number of masks ', num2str(params.N_mask)];
disp(resname);

%% Initialize result recording matrices
len_range = length(params.Range);
knn_acc = zeros(t_max, len_range); svm_acc = knn_acc; rf_acc = knn_acc;
knn_kappa = knn_acc; svm_kappa = knn_acc; rf_kappa = knn_acc;
knn_aa = knn_acc; svm_aa = knn_acc; rf_aa = knn_acc;
knn_f1 = knn_acc; svm_f1 = knn_acc; rf_f1 = knn_acc;
tim_run = knn_acc; tim_classify = knn_acc;

for t = 1:t_max
    tic
    % Data preprocessing
    [M1,S1] = preTrain(params,data{1},label{1},similarity_matrix);
    M1 = M1 + IE1;
    [M2,S2] = preTrain(params,data{2},label{2},similarity_matrix);
    M2 = M2 + IE2;
    Mask_sum = [M1; M2];
    Score_sum = [S1; S2];
    
    % Train the neural network
    t1 = toc;
    net = trainAndPredictNN(Mask_sum, Score_sum,hiddenSizes);
    t2 = toc;
    params.net = net;

    for N_size = 1:size(params.Range,2)
        tic
        N_sel = params.Range(N_size);
        % Adjust the grouping structure based on similarity
        final_groups = adjustGroups(similarity_matrix, N_sel);
        % Perform PSO-based band selection
        [selectedBand] = PSO(params, N_sel, final_groups);
        t3 = toc;
        disp(['Search Time (s): ', num2str(t3)]);
        
        % Subset the data using selected bands
        subdata = data3(:, selectedBand);
        
        % Perform classification using KNN, SVM, and RF
        [knn_acc(t,N_size), svm_acc(t,N_size), rf_acc(t,N_size), ...
         knn_kappa(t,N_size), svm_kappa(t,N_size), rf_kappa(t,N_size), ...
         knn_aa(t,N_size), svm_aa(t,N_size), rf_aa(t,N_size), ...
         knn_f1(t,N_size), svm_f1(t,N_size), rf_f1(t,N_size)] = classify(subdata, label3);
        t4 = toc;
        
        % Display all evaluation metrics
        fprintf('--- Number of Bands: %d ---\n', N_sel);
        fprintf('KNN   -> Accuracy: %.4f | Kappa: %.4f | AA: %.4f | F1: %.4f\n', ...
            knn_acc(t,N_size), knn_kappa(t,N_size), knn_aa(t,N_size), knn_f1(t,N_size));
        fprintf('SVM   -> Accuracy: %.4f | Kappa: %.4f | AA: %.4f | F1: %.4f\n', ...
            svm_acc(t,N_size), svm_kappa(t,N_size), svm_aa(t,N_size), svm_f1(t,N_size));
        fprintf('RF    -> Accuracy: %.4f | Kappa: %.4f | AA: %.4f | F1: %.4f\n', ...
            rf_acc(t,N_size), rf_kappa(t,N_size), rf_aa(t,N_size), rf_f1(t,N_size));
        fprintf('Running Time -> Search: %.2fs | Classify: %.2fs\n\n', t3-t2, t4 - t3);

        % Record timing information
        tim_run(t,N_size) = t1;
        tim_classify(t,N_size) = t2 - t1;
        tim_pre(t,N_size) = t1;
        tim_train(t,N_size) = t2 - t1;
        tim_run(t,N_size) = t3;
        tim_classify(t,N_size) = t4 - t3;
    end
end

%% Compute average results
knn_avg = [knn_acc; mean(knn_acc,1)];
svm_avg = [svm_acc; mean(svm_acc,1)];
rf_avg  = [rf_acc;  mean(rf_acc,1)];
knn_kappa = [knn_kappa; mean(knn_kappa,1)];
svm_kappa = [svm_kappa; mean(svm_kappa,1)];
rf_kappa  = [rf_kappa;  mean(rf_kappa,1)];
knn_aa = [knn_aa; mean(knn_aa,1)];
svm_aa = [svm_aa; mean(svm_aa,1)];
rf_aa  = [rf_aa;  mean(rf_aa,1)];
knn_f1 = [knn_f1; mean(knn_f1,1)];
svm_f1 = [svm_f1; mean(svm_f1,1)];
rf_f1  = [rf_f1;  mean(rf_f1,1)];

tim_run = [tim_run, sum(tim_run,2)];
tim_run(t_max+1,:) = mean(tim_run);
tim_classify = [tim_classify, sum(tim_classify,2)];
tim_classify(t_max+1,:) = mean(tim_classify);

%% Save all results
folderPath = ['RES/', data_type_t, '/'];
if exist(folderPath, 'dir') ~= 7
    mkdir(folderPath);
end

dateTag = datestr(now, 'yyyymmdd');

% Save each metric as a separate table
writetable(array2table(100*knn_avg, 'VariableNames', cellstr("Band_"+params.Range)), ...
    [folderPath, resname, '_KNN_Acc_', dateTag, '.csv']);
writetable(array2table(100*svm_avg, 'VariableNames', cellstr("Band_"+params.Range)), ...
    [folderPath, resname, '_SVM_Acc_', dateTag, '.csv']);
writetable(array2table(100*rf_avg, 'VariableNames', cellstr("Band_"+params.Range)), ...
    [folderPath, resname, '_RF_Acc_', dateTag, '.csv']);

writetable(array2table(knn_kappa), [folderPath, resname, '_KNN_Kappa_', dateTag, '.csv']);
writetable(array2table(svm_kappa), [folderPath, resname, '_SVM_Kappa_', dateTag, '.csv']);
writetable(array2table(rf_kappa),  [folderPath, resname, '_RF_Kappa_', dateTag, '.csv']);

writetable(array2table(knn_aa), [folderPath, resname, '_KNN_AA_', dateTag, '.csv']);
writetable(array2table(svm_aa), [folderPath, resname, '_SVM_AA_', dateTag, '.csv']);
writetable(array2table(rf_aa),  [folderPath, resname, '_RF_AA_', dateTag, '.csv']);

writetable(array2table(knn_f1), [folderPath, resname, '_KNN_F1_', dateTag, '.csv']);
writetable(array2table(svm_f1), [folderPath, resname, '_SVM_F1_', dateTag, '.csv']);
writetable(array2table(rf_f1),  [folderPath, resname, '_RF_F1_', dateTag, '.csv']);

writetable(array2table(tim_run),     [folderPath, resname, '_RunTime_', dateTag, '.csv']);
writetable(array2table(tim_classify),[folderPath, resname, '_ClassifyTime_', dateTag, '.csv']);
