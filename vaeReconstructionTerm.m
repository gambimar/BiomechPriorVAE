function output = vaeReconstructionTerm(obj, option, X, vaeParams)

fctname = 'vaeReconstructionTerm';

if strcmp(option,'init')
    
    if ~isfield(obj.idx,'states')
        error('Model states are not stored in state vector X.')
    end
    modulePath = which('vaemodel.py');
    if isempty(modulePath)
        error('Add the BiomechPriorVAE src directory to the MATLAB path.');
    end
    sourceDir = fileparts(modulePath);
    repositoryDir = fileparts(sourceDir);
    if isfield(vaeParams, 'pythonExecutable') && ~isempty(vaeParams.pythonExecutable)
        pythonExecutable = vaeParams.pythonExecutable;
    elseif ispc
        pythonExecutable = fullfile(repositoryDir, '.venv', 'Scripts', 'python.exe');
    else
        pythonExecutable = fullfile(repositoryDir, '.venv', 'bin', 'python');
    end
    if ~isfile(pythonExecutable)
        error('Python interpreter not found: %s. Set vaeParams.pythonExecutable.', pythonExecutable);
    end
    obj.objectiveInit.(fctname).pyenv = pyenv(Version=pythonExecutable, ExecutionMode="OutOfProcess");
    if int64(py.sys.path.count(repositoryDir)) == 0
        insert(py.sys.path, int64(0), repositoryDir);
    end
    if int64(py.sys.path.count(sourceDir)) == 0
        insert(py.sys.path, int64(0), sourceDir);
    end

    [currentRows, momentOrder] = vaeStateIndices(obj.model);
    obj.objectiveInit.(fctname).currentIdx = obj.idx.states(currentRows, 1:obj.nNodes);
    obj.objectiveInit.(fctname).momentOrder = momentOrder;
    obj.objectiveInit.(fctname).vaeParams = vaeParams;
    [useMoments, stateKeys] = vaeFeatureLayout(vaeParams);
    obj.objectiveInit.(fctname).useMoments = useMoments;
    if ~isempty(stateKeys)
        pythonStateKeys = py.tuple(stateKeys);
    else
        pythonStateKeys = py.None;
    end
    if obj.objectiveInit.(fctname).useMoments
        if isfield(vaeParams, 'momentScale')
            momentScale = vaeParams.momentScale;
        else
            momentScale = 1 / (9.81 * obj.model.bodymass * obj.model.bodyheight);
        end
        validateattributes(momentScale, {'numeric'}, {'scalar', 'finite', 'positive'});
    else
        momentScale = 0;
    end
    obj.objectiveInit.(fctname).momentScale = momentScale;
    
    try
        %py.sys.path().append(vaeParams.pythonPath);
        obj.objectiveInit.(fctname).vaeModule = py.importlib.import_module('vaemodel');
        
        success = obj.objectiveInit.(fctname).vaeModule.initialize_vae(...
            vaeParams.modelPath, ...
            vaeParams.scalerPath, ...
            pyargs('num_dofs', int32(vaeParams.numDofs), ...
                   'latent_dim', int32(vaeParams.latentDim), ...
                   'hidden_dim', int32(vaeParams.hiddenDim), ...
                   'device', vaeParams.device, ...
                   'state_keys', pythonStateKeys));
        
        if ~success
            error('Failed to initialize VAE model');
        end
        
    catch ME
        error('Failed to initialize Python VAE interface: %s', ME.message);
    end
    
    obj.objectiveInit.(fctname).nJoints = length(obj.model.extractState('q'));
    
    output = NaN;
    return;
end

currentIdx = obj.objectiveInit.(fctname).currentIdx;
vaeModule = obj.objectiveInit.(fctname).vaeModule;
useMoments = obj.objectiveInit.(fctname).useMoments;
mom_scale_fac = obj.objectiveInit.(fctname).momentScale;
currentJoints = X(currentIdx);

[M, dMdx, dUdx] = vaeJointMoments(obj.model, X(obj.idx.states), X(obj.idx.controls), ...
    useMoments, strcmp(option, 'gradient'), obj.objectiveInit.(fctname).momentOrder);
currentJoints = [currentJoints; M*mom_scale_fac];

if strcmp(option,'objval')
    pyJoints = py.numpy.array(currentJoints);
    pyResult = vaeModule.reconstruct(pyJoints);
    output = double(pyResult);
elseif strcmp(option,'gradient')
    pyJoints = py.numpy.array(currentJoints);
    pyResult = vaeModule.reconstruct_withgrad(pyJoints);
    gradient = double(pyResult);
    output = vaeStateGradient(X, gradient, currentIdx, obj.idx, mom_scale_fac, dMdx, dUdx, useMoments);
else
    error('vaeReconstructionTerm:InvalidOption', 'Unknown VAE objective option: %s', option);
end
end
