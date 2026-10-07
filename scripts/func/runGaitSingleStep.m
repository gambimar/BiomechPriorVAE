function done = runGaitSingleStep(speed, prefix, model_file, metmodel)
    % Set up paths so that the script can run from any folder
    addpath(genpath("/home/rzlin/ri94mihu/phd/BiomechPriorVAE/"));
    addpath(genpath("/home/rzlin/ri94mihu/phd/mexIPOPT"));

    % set up matlabs rng generator

    base_result_path = "/home/rzlin/ri94mihu/phd/BiomechPriorVAE/result/simulations/";
    n_conv = 0;
    iter = 1;
    max_iter = 10;
    if speed < 0.01
        max_iter = 10;
    end
    while n_conv < max_iter
        rng(iter)
        name = strcat(base_result_path, prefix, string(iter), "_", string(speed), ".mat");
        if ~exist(name,'file')
            if nargin < 4 || isempty(metmodel) || strcmp(metmodel, '0')
                runSim(speed, name, model_file);
            else
                runSim(speed, name, model_file, metmodel);
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
