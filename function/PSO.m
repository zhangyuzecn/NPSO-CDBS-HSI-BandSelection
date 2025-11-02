function [selectedBand] = PSO(params, N_sel, final_groups)

    % Extract parameters
    net = params.net;
    IE = params.IE;
    N_bands = params.N_bands;
    N_ind = params.N_ind;
    N_task = params.N_task;
    GX = params.GX;

    % PSO coefficient for cognitive and social components
    C = 0.49445;

    % Total number of particles in the population
    N_pop = N_task * N_ind;

    %----------------------------------------------------------------------
    % Initialize population
    %----------------------------------------------------------------------
    S_band = calBandScore(N_bands, net);              % Calculate band importance score
    X = popInitialize(N_pop, N_bands, N_sel, S_band); % Initialize positions (particles)
    V = 0.5 * rand(N_pop, N_bands);                   % Initialize velocities randomly

    % Velocity boundaries
    vmax = 0.5;
    vmin = -vmax;

    % Initialize personal and global bests
    pbest = X;
    for j = 1:N_ind
        obj_P(j,:) = fitness(net, X(j,:), N_bands, final_groups, IE);
    end
    obj_pbest = obj_P;

    % Find initial global best solution
    idx_gbest1 = findgbest(obj_pbest(1:N_ind,:));
    gbest(1,:) = X(idx_gbest1,:);
    obj_gbest(1,:) = obj_pbest(idx_gbest1,:);

    %----------------------------------------------------------------------
    % Main optimization loop
    %----------------------------------------------------------------------
    UP = 0;
    for t = 1:GX

        % Update inertia weight linearly decreasing from 0.9 to 0.4
        W = 0.9 - 0.5 * (t / GX);

        % Iterate over each individual particle
        for j = 1:N_ind

            %----------------------------
            % Velocity update (PSO core)
            %----------------------------
            V_new = W * V(j,:) ...
                  + C * rand * (pbest(j,:) - X(j,:)) ...
                  + C * rand * (gbest - X(j,:));

            %----------------------------
            % Velocity constraint
            %----------------------------
            for k = 1:N_bands
                if V_new(k) > vmax
                    V_new(k) = vmax;
                elseif V_new(k) < vmin
                    V_new(k) = vmin;
                end
            end

            %----------------------------
            % Position update
            %----------------------------
            V(j,:) = V_new;
            X(j,:) = X(j,:) + V(j,:);

            %----------------------------
            % Position constraint (keep within [0,1])
            %----------------------------
            for k = 1:N_bands
                if X(k) > 1
                    if rand < 0.5
                        X(k) = 1;
                        V(k) = -V(k);
                    else
                        X(k) = rand;
                        V(k) = rand * 0.5;
                    end
                end
                if X(k) <= 0
                    if rand < 0.5
                        X(k) = 0;
                        V(k) = -V(k);
                    else
                        X(k) = rand;
                        V(k) = rand * 0.5;
                    end
                end
            end

            %----------------------------
            % Fitness evaluation
            %----------------------------
            obj_new = fitness(net, X(j,:), N_bands, final_groups, IE);

            %----------------------------
            % Update personal best
            %----------------------------
            if obj_new < obj_pbest(j,:)
                obj_pbest(j,:) = obj_new;
                pbest(j,:) = X(j,:);
            end

            %----------------------------
            % Update global best
            %----------------------------
            if obj_new < obj_gbest
                obj_gbest = obj_new;
                gbest = X(j,:);
                UP = 0;
            end
        end

        % Display iteration information
        UP = UP + 1;
        disp(['Current iteration: ', num2str(t), ...
              '  Fitness: ', num2str(obj_gbest)]);
    end

    %----------------------------------------------------------------------
    % Decode best particle to obtain selected band indices
    %----------------------------------------------------------------------
    pop = gbest(1,:);
    selectedBand = decoding(pop, final_groups);
end
