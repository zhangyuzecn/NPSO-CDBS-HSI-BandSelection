function entropyValues = computeBandEntropy(inputData)

    % Number of histogram bins
    numBins = 256;
    
    % Get size of input data
    [numBands, numSamples] = size(inputData);
    
    % Initialize entropy vector
    entropyPerBand = zeros(numBands, 1);
    
    % Define histogram edges based on data range
    minVal = min(inputData(:));
    maxVal = max(inputData(:));
    binEdges = linspace(minVal, maxVal, numBins);
    
    % Compute entropy for each band
    for bandIdx = 1:numBands
        % Histogram normalized to probability distribution
        histValues = hist(inputData(bandIdx, :), binEdges) / numSamples;
        
        % Compute entropy (add eps to avoid log(0))
        entropyPerBand(bandIdx) = - histValues * log(histValues + eps)';
    end
    
    % Rank entropy values in descending order
    [sortedEntropy, sortedIdx] = sort(entropyPerBand, 'descend');
    for i = 1:numBands
        entropyPerBand(sortedIdx(i)) = sortedEntropy(i);
    end    
    
    % Normalize entropy values to [0,1]
    entropyValues = entropyPerBand';
    entropyValues = (entropyValues - min(entropyValues)) / ...
                    (max(entropyValues) - min(entropyValues));
end
