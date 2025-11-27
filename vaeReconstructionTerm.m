function output = vaeReconstructionTerm(obj, option, X, vaeParams)

fctname = 'vaeReconstructionTerm';

if strcmp(option,'init')
    
    if ~isfield(obj.idx,'states')
        error('Model states are not stored in state vector X.')
    end
    obj.objectiveInit.(fctname).pyenv = pyenv(Version="/home/rzlin/ri94mihu/phd/BiomechPriorVAE/.venv/bin/python3.12", ExecutionMode="OutOfProcess");
    insert(py.sys.path, int64(0), what("BiomechPriorVAE").path);
    insert(py.sys.path, int64(0), strcat(what("BiomechPriorVAE").path,filesep,"src"));

    obj.objectiveInit.(fctname).idxJointsAllNodes = obj.idx.states(:, 1:obj.nNodes);
    obj.objectiveInit.(fctname).vaeParams = vaeParams;
    
    try
        %py.sys.path().append(vaeParams.pythonPath);
        obj.objectiveInit.(fctname).vaeModule = py.importlib.import_module('vaemodel');
        
        success = obj.objectiveInit.(fctname).vaeModule.initialize_vae(...
            vaeParams.modelPath, ...
            vaeParams.scalerPath, ...
            pyargs('num_dofs', int32(vaeParams.numDofs), ...
                   'latent_dim', int32(vaeParams.latentDim), ...
                   'hidden_dim', int32(vaeParams.hiddenDim), ...
                   'device', vaeParams.device));
        
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

idxJointsAllNodes = obj.objectiveInit.(fctname).idxJointsAllNodes;
vaeModule = obj.objectiveInit.(fctname).vaeModule;
end_ = obj.model.nStates;
currentIdx = idxJointsAllNodes([7:33 40:66 251:6:304 252:6:304 253:6:304 305:6:end_ 306:6:end_ 307:6:end_], 1:obj.nNodes);
currentJoints = X(currentIdx);

x_ = X(obj.idx.states);
M = zeros(obj.nNodes, 33);
dMdx = {};
for i = 1:obj.nNodes
    [M_,dM_,dU_] = obj.model.getJointmoments(x_(:,i));
    if i == 1
        dMdx = {dM_};
        dUdx = {dU_};
    else
        dMdx = [dMdx, {dM_}];
        dUdx = [dUdx, {dU_}];
    end
    M(i,:) = M_';
end
% Hardcoding mass and height here
mom_scale_fac = 1/9.81/1.83/65.9; % Hamner default values
currentJoints = [currentJoints; M'*mom_scale_fac];

if strcmp(option,'objval')
    pyJoints = py.numpy.array(currentJoints);
    pyResult = vaeModule.reconstruct(pyJoints);
    nodeError = double(pyResult);
    output = nodeError;

    
elseif strcmp(option,'gradient')
    output = zeros(size(X));
    pyJoints = py.numpy.array(currentJoints);
    pyResult = vaeModule.reconstruct_withgrad(pyJoints);
    gradient = double(pyResult);
    dVAEdx = gradient(:,1:102);
    output(currentIdx) = dVAEdx';
    dVEAdM = gradient(:,103:end);
    for i = 1:obj.nNodes
        output(obj.idx.states(:,i)) = output(obj.idx.states(:,i)) + dMdx{i} * dVEAdM(i,:)'*mom_scale_fac;
        output(obj.idx.controls(:,i)) = output(obj.idx.controls(:,i)) + dUdx{i} * dVEAdM(i,:)'*mom_scale_fac;
    end
end

end