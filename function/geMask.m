function mask = geMask(group_indices)
    % Initialize a zero mask of length 176
    mask = zeros(1, 176);

    % Initialize an empty list for selected indices
    selected_indices = [];

    % Randomly select one position from each group interval
    for i = 1:length(group_indices) - 1
        start_idx = group_indices(i);
        end_idx = group_indices(i + 1);

        % Ensure the range is valid
        if end_idx >= start_idx
            current_range = start_idx:end_idx;
            % Randomly pick one index from the current group
            selected_indices(end + 1) = current_range(randi(numel(current_range)));
        end
    end

    % Set selected positions to 1
    mask(selected_indices) = 1;
end
