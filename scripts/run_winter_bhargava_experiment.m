function run_winter_bhargava_experiment()
    % Setup paths
    addpath(genpath('/home/rzlin/ri94mihu/phd/BiomechPriorVAE/'));
    addpath(genpath('/home/rzlin/ri94mihu/phd/mexIPOPT'));

    % Parameters
    modelFile = 'sipp_generic_runmad_smoothsphere.osim';
    metmodel  = 'bhargavaact';
    targetspeed = 1.33;
    N = 50;
    sym = 1;
    num_runs = 10;
    noise_std = 0.02; % ~1.15 degrees noise std

    % Paths
    path2repo = '/home/rzlin/ri94mihu/phd/BiomechPriorVAE';
    outDir = [path2repo filesep 'result' filesep 'simulations' filesep 'winter_bhargava'];
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    % 1. Load Winter 1.33 m/s data
    winterFile = which('Winter_normal.mat');
    if isempty(winterFile)
        error('Winter_normal.mat not found in MATLAB path!');
    end
    wData = load(winterFile);
    wVars = wData.dataStruct.variables;
    fprintf('Loaded Winter_normal.mat successfully.\n');

    % 2. Standing problem file
    resultFileStanding = [outDir filesep 'standing.mat'];
    model = Gait3d_smoothsphere(modelFile);
    if ~exist(resultFileStanding, 'file')
        problemStanding = standing3D(model, resultFileStanding);
        solverStanding = IPOPT();
        solverStanding.setOptionField('tol', 1e-4);
        solverStanding.setOptionField('constr_viol_tol', 1e-3);
        solverStanding.setOptionField('dual_inf_tol', 1e-4);
        solverStanding.setOptionField('acceptable_tol', 1e-4);
        solverStanding.setOptionField('print_level', 0);
        resultStanding = solverStanding.solve(problemStanding);
        resultStanding.save(resultFileStanding);
        fprintf('Standing problem solved and saved to %s\n', resultFileStanding);
    else
        fprintf('Loaded existing standing file: %s\n', resultFileStanding);
    end

    % Objective weights
    W.effMuscles = 0;
    W.effMusclesAct = 0.0;
    W.effMusclesTor = 1;
    W.metcost = 1;
    W.effTorques = 0;
    W.reg = 0;
    W.track = 0;
    W.dur = 0;
    W.speed = Inf;
    W.heelstrikeVariance = 100;

    % VAE Params
    vaeParams.modelPath  = which('BiomechPriorVAE_best_50.pth');
    vaeParams.scalerPath = which('scaler_50.pkl');
    vaeParams.pythonPath = fileparts(which('runSim.m'));
    vaeParams.numDofs    = 50;
    vaeParams.latentDim  = 24;
    vaeParams.hiddenDim  = 512;
    vaeParams.device     = 'cpu';
    vaeParams.weight     = 1;

    % Winter joint angle names present in dataStruct
    winterAngleNames = {'hip_flexion_r', 'knee_angle_r', 'ankle_angle_r', ...
                        'hip_flexion_l', 'knee_angle_l', 'ankle_angle_l'};

    runtimes = zeros(num_runs, 1);
    converged = false(num_runs, 1);
    iterations = zeros(num_runs, 1);
    final_objs = zeros(num_runs, 1);

    fprintf('\n========================================================\n');
    fprintf('Starting 10 simulations: model=%s, metmodel=%s, speed=%.2f\n', modelFile, metmodel, targetspeed);
    fprintf('Initial guess: Winter 1.33 m/s joint angles + Gaussian noise (std=%.3f rad)\n', noise_std);
    fprintf('========================================================\n\n');

    for run_idx = 1:num_runs
        runResultFile = sprintf('%s/sim_run_%02d.mat', outDir, run_idx);

        if exist(runResultFile, 'file')
            fprintf('Run %02d/10 already completed. Loading from file...\n', run_idx);
            loaded = load(runResultFile);
            res = loaded.result;
            converged(run_idx) = res.converged;
            if isfield(res.info, 'wallTime')
                runtimes(run_idx) = res.info.wallTime;
            elseif isfield(res.info, 'cpu')
                runtimes(run_idx) = res.info.cpu;
            end
            if isfield(res.info, 'iter')
                iterations(run_idx) = res.info.iter;
            end
            if isfield(res.info, 'objective')
                final_objs(run_idx) = res.info.objective;
            end
            fprintf('Run %02d/10 (Cached): runtime = %7.2f s | converged = %d | iter = %d | objective = %g\n', ...
                    run_idx, runtimes(run_idx), converged(run_idx), iterations(run_idx), final_objs(run_idx));
            continue;
        end

        % Build problem instance
        prob = running3D(model, 1, resultFileStanding, runResultFile, N, sym, W, targetspeed, vaeParams, metmodel);

        % Apply perturbed Winter joint angles to initial guess
        rng(run_idx);
        for iv = 1:length(winterAngleNames)
            varName = winterAngleNames{iv};
            idxInTable = strmatch(varName, wVars.name, 'exact');
            if ~isempty(idxInTable)
                meanVal = wVars.mean{idxInTable}; % 50x1
                % Perturb with noise
                perturbedVal = meanVal + noise_std * randn(size(meanVal));

                state_idx = model.extractState('q', varName);
                for nodeIdx = 1:N
                    idxInX = prob.idx.states(state_idx, nodeIdx);
                    prob.initialguess.X(idxInX) = perturbedVal(nodeIdx);
                end
            end
        end

        % Configure IPOPT
        solver = IPOPT();
        solver.setOptionField('max_iter', 10000);
        solver.setOptionField('tol', 1e-3);
        solver.setOptionField('print_level', 5);
        solver.setOptionField('dual_inf_tol', 1e-3);
        solver.setOptionField('constr_viol_tol', 1e-4);

        % Solve and time
        t_start = tic;
        res = solver.solve(prob);
        t_elapsed = toc(t_start);

        if isfield(res.info, 'wallTime') && res.info.wallTime > 0
            runtimes(run_idx) = res.info.wallTime;
        else
            runtimes(run_idx) = t_elapsed;
        end
        converged(run_idx) = res.converged;
        if isfield(res.info, 'iter')
            iterations(run_idx) = res.info.iter;
        end
        if isfield(res.info, 'objective')
            final_objs(run_idx) = res.info.objective;
        end

        res.save(runResultFile);

        fprintf('Run %02d/10: runtime = %7.2f s | converged = %d | iter = %d | objective = %g\n', ...
                run_idx, runtimes(run_idx), converged(run_idx), iterations(run_idx), final_objs(run_idx));
    end

    % Summary Report
    fprintf('\n========================================================\n');
    fprintf('BENCHMARK SUMMARY REPORT\n');
    fprintf('========================================================\n');
    fprintf('Model:                %s\n', modelFile);
    fprintf('Metabolic Model:      %s\n', metmodel);
    fprintf('Speed Bins:           %.2f m/s\n', targetspeed);
    fprintf('Total Runs:           %d\n', num_runs);
    fprintf('Converged Runs:       %d / %d\n', sum(converged), num_runs);
    fprintf('Runtimes per run (s): %s\n', mat2str(round(runtimes, 2)'));
    fprintf('Mean Runtime:         %.2f s (std: %.2f s)\n', mean(runtimes), std(runtimes));
    fprintf('Min Runtime:          %.2f s\n', min(runtimes));
    fprintf('Max Runtime:          %.2f s\n', max(runtimes));
    fprintf('Total Runtime:        %.2f s (%.2f min)\n', sum(runtimes), sum(runtimes)/60);
    fprintf('========================================================\n');

    % Save benchmark summary file
    summaryFile = [outDir filesep 'benchmark_summary.mat'];
    save(summaryFile, 'runtimes', 'converged', 'iterations', 'final_objs', 'modelFile', 'metmodel', 'targetspeed', 'noise_std');
    fprintf('Summary saved to %s\n', summaryFile);
end
