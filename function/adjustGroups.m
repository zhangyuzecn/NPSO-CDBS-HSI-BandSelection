function final_groups = adjustGroups(similarity_matrix, N_Group)
    % adjustGroups - Adjust group boundaries based on band similarity
    %
    % Inputs:
    %   similarity_matrix : (num_bands x num_bands) band similarity matrix
    %   N_Group            : desired number of groups
    %
    % Output:
    %   final_groups       : indices defining the adjusted group boundaries

    % Total number of spectral bands
    num_bands = size(similarity_matrix, 1);

    % Initialize group boundaries (uniform division)
    group_idx = zeros(N_Group + 1, 1);
    for i = 1:N_Group
        group_idx(i) = floor(num_bands / N_Group * (i - 1)) + 1;
    end
    group_idx(N_Group + 1) = num_bands;

    % Iteratively adjust group boundaries based on similarity continuity
    for i = 1:N_Group - 1
        flag = true;
        iter_count = 0;

        while flag
            % Current and next group index ranges
            current_group = group_idx(i):group_idx(i + 1) - 1;
            next_group = group_idx(i + 1):group_idx(i + 2);

            % Mean similarity between last band of current group and its own group
            sim_curr_end_curr = mean(similarity_matrix(current_group, group_idx(i + 1) - 1));
            % Mean similarity between last band of current group and next group
            sim_curr_end_next = mean(similarity_matrix(next_group, group_idx(i + 1) - 1));

            % Condition A: current group's end is more similar to next group
            A = sim_curr_end_curr < sim_curr_end_next;

            % Mean similarity between first band of next group and both groups
            sim_next_start_curr = mean(similarity_matrix(current_group, group_idx(i + 1)));
            sim_next_start_next = mean(similarity_matrix(next_group, group_idx(i + 1)));

            % Condition B: next group's start is more similar to next group
            B = sim_next_start_curr < sim_next_start_next;

            % Sum of conditions determines boundary adjustment
            sum_AB = A + B;

            if sum_AB == 2
                % Both conditions indicate continuity; shift boundary right
                group_idx(i + 1) = group_idx(i + 1) + 1;
            elseif sum_AB == 0
                % Both conditions indicate separation; shift boundary left
                group_idx(i + 1) = group_idx(i + 1) - 1;
            end

            % Stop adjusting if stable (A and B disagree)
            if sum_AB == 1
                flag = false;
            end

            iter_count = iter_count + 1;
            if iter_count >= floor(num_bands / (N_Group * 2))
                break; % Avoid infinite loop
            end
        end
    end

    % Output final group boundaries
    final_groups = group_idx;
end
