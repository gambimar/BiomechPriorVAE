function run_single_winter_bhargava(run_idx)
    % Setup paths
    addpath(genpath('/home/rzlin/ri94mihu/phd/BiomechPriorVAE/'));
    addpath(genpath('/home/rzlin/ri94mihu/phd/mexIPOPT'));

    modelFile = 'sipp_generic_runmad_smoothsphere.osim';
    metmodel  = 'bhargavaact';
    targetspeed = 1.33;
    N = 50;
    sym = 1;
    noise_std = 0.02;

    path2repo = '/home/rzlin/ri94mihu/phd/BiomechPriorVAE';
    outDir = [path2repo filesep 'result' filesep 'simulations' filesep 'winter_bhargava'];
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    runResultFile = sprintf('%s/sim_run_%02d.mat', outDir, run_idx);
    if exist(runResultFile, 'file')
        fprintf('Run %02d already exists. Skipping.\n', run_idx);
        return;
    end

    winterFile = which('Winter_normal.mat');
    wData = load(winterFile);
    wVars = wData.dataStruct.variables;

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
    end

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

    vaeParams.modelPath  = which('BiomechPriorVAE_best_50.pth');
    vaeParams.scalerPath = which('scaler_50.pkl');
    vaeParams.pythonPath = fileparts(which('runSim.m'));
    vaeParams.numDofs    = 50;
    vaeParams.latentDim  = 24;
    vaeParams.hiddenDim  = 512;
    vaeParams.device     = 'cpu';
    vaeParams.weight     = 1;

    winterAngleNames = {'hip_flexion_r', 'knee_angle_r', 'ankle_angle_r', ...
                        'hip_flexion_l', 'knee_angle_l', 'ankle_angle_l'};

    prob = running3D(model, 1, resultFileStanding, runResultFile, N, sym, W, targetspeed, vaeParams, metmodel);

    rng(run_idx);
    for iv = 1:length(winterAngleNames)
        varName = winterAngleNames{iv};
        idxInTable = strmatch(varName, wVars.name, 'exact');
        if ~isempty(idxInTable)
            meanVal = wVars.mean{idxInTable};
            perturbedVal = meanVal + noise_std * randn(size(meanVal));
            state_idx = model.extractState('q', varName);
            for nodeIdx = 1:N
                idxInX = prob.idx.states(state_idx, nodeIdx);
                prob.initialguess.X(idxInX) = perturbedVal(nodeIdx);
            end
        end
    end

    solver = IPOPT();
    solver.setOptionField('max_iter', 10000);
    solver.setOptionField('tol', 1e-3);
    solver.setOptionField('print_level', 5);
    solver.setOptionField('dual_inf_tol', 1e-3);
    solver.setOptionField('constr_viol_tol', 1e-4);

    t_start = tic;
    res = solver.solve(prob);
    t_elapsed = toc(t_start);

    res.save(runResultFile);
    fprintf('Run %02d completed: wallTime = %.2f s | converged = %d | iter = %d | objective = %g\n', ...
            run_idx, t_elapsed, res.converged, res.info.iter, res.info.objective);
end
