function selectedBand = decoding(X, group_idx)
    % Number of groups
    numGroups = length(group_idx) - 1;

    % Initialize array to store the index of the maximum value in each group
    maxIndices = zeros(1, numGroups);

    % Find the index of the maximum element in each group
    for i = 1:numGroups
        current_group = group_idx(i):group_idx(i + 1) - 1;
        [~, maxLocalIdx] = max(X(current_group));
        maxIndices(i) = current_group(maxLocalIdx);  % Convert to global index
    end

    % Return the selected band indices
    selectedBand = maxIndices;
end
