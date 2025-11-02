function net = trainAndPredictNN(X_train, y_train, hiddenSizes)
    trainFcn = 'trainscg';     % Training algorithm: Scaled Conjugate Gradient
    learningRate = 0.01;       % Learning rate
    maxEpochs = 1000;          % Maximum number of training epochs
    batchSize = 32;            % Mini-batch size
    regularization = 0.01;     % L2 regularization coefficient
    % stopCondition = 0.0001;  % Optional stop condition (error threshold)

    % Create a feedforward neural network
    net = feedforwardnet(hiddenSizes); % e.g., [10] means one hidden layer with 10 neurons

    net.trainFcn = trainFcn;                    % Training algorithm
    net.trainParam.lr = learningRate;           % Learning rate
    net.trainParam.epochs = maxEpochs;          % Max training iterations
    net.trainParam.batchSize = batchSize;       % Mini-batch size
    net.trainParam.max_fail = 100;              % Early stopping patience (validation check)
    net.trainParam.goal = 0.001;                % Training goal (performance target)
    net.performParam.regularization = regularization; % Regularization weight

    % Set the output layer activation function
    net.layers{end}.transferFcn = 'logsig';     % Logistic sigmoid function

    % Transpose input and output to match MATLAB Neural Network Toolbox format
    X_train = X_train';   
    y_train = y_train';   

    net = train(net, X_train, y_train);

end
