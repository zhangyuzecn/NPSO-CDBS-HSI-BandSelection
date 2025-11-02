function knnAccuracy = classify_gpu_KNN(bands, data, label)

    % Select data from chosen bands
    data = data(:, bands);

    % Split dataset into training and testing sets (80% holdout)
    cv = cvpartition(size(data, 1), 'HoldOut', 0.8);
    trainData  = data(cv.training, :);
    testData   = data(cv.test, :);
    trainLabel = label(cv.training, :);
    testLabel  = label(cv.test, :);

    % Move data to GPU if available (fitcknn itself runs on CPU)
    trainData_gpu = gpuArray(trainData);
    testData_gpu  = gpuArray(testData);

    % Convert labels to GPU arrays for consistency
    trainLabel_gpu = gpuArray(trainLabel);
    testLabel_gpu  = gpuArray(testLabel);

    % Train KNN model on selected bands (executed on CPU)
    % Note: fitcknn does not support full GPU training, but data transfer helps pre-processing
    knnModel = fitcknn(gather(trainData_gpu), gather(trainLabel_gpu), 'NumNeighbors', 5);

    % Perform prediction and compute accuracy
    knnPredictions = predict(knnModel, gather(testData_gpu));
    knnAccuracy = mean(knnPredictions == gather(testLabel_gpu));
end
