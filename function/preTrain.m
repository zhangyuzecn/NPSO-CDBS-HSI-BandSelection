function [Mask, Score] = preTrain(params, data, label, similarity_matrix)

    range = params.Range;  % Range of band subset sizes to test

    for k = 1:numel(range)
        % Adjust groups according to similarity threshold
        final_groups = adjustGroups(similarity_matrix, range(k));

        score_i = zeros(params.N_mask, 1);  % Initialize score list
        mask_i = zeros(params.N_mask, params.N_bands);  % Initialize mask storage

        % Generate random masks and evaluate using KNN classifier
        for i = 1:params.N_mask
            mask_i(i, :) = geMask(final_groups);

            % Select top bands according to the generated mask
            [~, sorted_idx] = sort(mask_i(i, :), 'descend');
            selected_bands = sorted_idx(1:range(k));

            % Compute classification accuracy for selected bands
            score_i(i) = classify_gpu_KNN(selected_bands, data, label);
        end

        Mask_cell{k} = mask_i;
        Score_cell{k} = score_i;
        disp(['Training data obtained for ', num2str(range(k)), ' bands.']);
    end

    % Concatenate results from all ranges
    Mask = vertcat(Mask_cell{:});
    Score = vertcat(Score_cell{:});
end
