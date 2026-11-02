function done = runGaitSingleStepVaeOff(speed, prefix, model_file, metmodel)
    % Set up paths so that the script can run from any folder
    path2repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    addpath(genpath(path2repo));
    addpath(genpath(fullfile(fileparts(path2repo), 'mexIPOPT')));

    % set up matlabs rng generator

    base_result_path = fullfile(path2repo, 'result', 'simulations');
    if ~isfolder(base_result_path)
        mkdir(base_result_path);
    end
    n_conv = 0;
    iter = 1;
    max_iter = 10;
    if speed < 0.01
        max_iter = 30;
    end
    while n_conv < max_iter
        rng(iter+30)
        name = fullfile(base_result_path, strcat(prefix,'noVae', string(iter), "_", string(speed), ".mat"));
        if ~exist(name,'file')
            if nargin < 4 || isempty(metmodel) || strcmp(metmodel, '0')
                runSimNoVae(speed, name, model_file);
            else
                runSimNoVae(speed, name, model_file, metmodel);
            end
            done = 0;
            return;
        end
        data = load(name);
        n_conv = n_conv + data.result.converged;
        iter = iter + 1;
    end
    fprintf('CONVERGED_10_REACHED_FOR_SPEED: %g\n', speed);
    done = 1;
end
