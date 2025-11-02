function [knnAccuracy, svmAccuracy, rfAccuracy, ...
          knnKappa, svmKappa, rfKappa, ...
          knnAA, svmAA, rfAA, ...
          knnF1, svmF1, rfF1] = classify(data, label)
    % classify - Evaluate classification performance using KNN, SVM, and RF
    %
    % Inputs:
    %   data  : feature matrix (samples × features)
    %   label : ground truth labels (samples × 1)
    %
    % Outputs:
    %   knnAccuracy, svmAccuracy, rfAccuracy : overall classification accuracy
    %   knnKappa, svmKappa, rfKappa          : Cohen's kappa coefficients
    %   knnAA, svmAA, rfAA                   : average accuracy (mean per-class)
    %   knnF1, svmF1, rfF1                   : mean F1-scores for each classifier

    % ======================= Data Partition =======================
    cv = cvpartition(size(data, 1), 'HoldOut', 0.9);
    trainData  = data(cv.training, :);
    testData   = data(cv.test, :);
    trainLabel = label(cv.training, :);
    testLabel  = label(cv.test, :);

    % ======================= Initialize =======================
    classifiers = {'KNN', 'SVM', 'RF'};
    predictions = cell(3, 1);
    accuracy = zeros(3, 1);
    kappa = zeros(3, 1);
    aa = zeros(3, 1);
    f1 = zeros(3, 1);

    % ======================= KNN =======================
    knnModel = fitcknn(trainData, trainLabel, 'NumNeighbors', 5);
    predictions{1} = predict(knnModel, testData);

    % ======================= SVM =======================
    svmTemplate = templateSVM('KernelFunction', 'rbf');
    svmModel = fitcecoc(trainData, trainLabel, 'Learners', svmTemplate);
    predictions{2} = predict(svmModel, testData);

    % ======================= Random Forest =======================
    rfModel = TreeBagger(50, trainData, trainLabel, 'Method', 'classification', 'OOBPrediction', 'off');
    predictions{3} = str2double(predict(rfModel, testData));

    % ======================= Metric Calculation =======================
    for i = 1:3
        pred = predictions{i};
        accuracy(i) = mean(pred == testLabel);

        % Confusion matrix
        cm = confusionmat(testLabel, pred);
        n = sum(cm(:));
        po = sum(diag(cm)) / n;
        pe = sum(sum(cm, 1) .* sum(cm, 2)') / (n^2);
        kappa(i) = (po - pe) / (1 - pe);

        % Average Accuracy (AA)
        classAcc = diag(cm) ./ sum(cm, 2);
        classAcc(isnan(classAcc)) = 0;
        aa(i) = mean(classAcc);

        % F1-score
        precision = diag(cm) ./ sum(cm, 1)';
        recall = diag(cm) ./ sum(cm, 2);
        precision(isnan(precision)) = 0;
        recall(isnan(recall)) = 0;
        f1_score = 2 * (precision .* recall) ./ (precision + recall);
        f1_score(isnan(f1_score)) = 0;
        f1(i) = mean(f1_score);
    end

    % ======================= Assign Outputs =======================
    knnAccuracy = accuracy(1); svmAccuracy = accuracy(2); rfAccuracy = accuracy(3);
    knnKappa    = kappa(1);    svmKappa    = kappa(2);    rfKappa    = kappa(3);
    knnAA       = aa(1);       svmAA       = aa(2);       rfAA       = aa(3);
    knnF1       = f1(1);       svmF1       = f1(2);       rfF1       = f1(3);

    % ======================= Optional Display =======================
    % for i = 1:3
    %     fprintf('%s -> Acc: %.2f%% | Kappa: %.4f | AA: %.4f | F1: %.4f\n', ...
    %         classifiers{i}, accuracy(i)*100, kappa(i), aa(i), f1(i));
    % end
end
