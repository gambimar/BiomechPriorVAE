%======================================================================
%> @file FAME\MyImplementationFC\CODrunning3DwthMeasuredIMU.m
%> @brief Function to specify the optimization problem for 3D COD
%running tracking measured IMU data
%>
%> @author Maria Eleni Athanasiadou
%> @date October, 2024
%======================================================================

% ======================================================================
%> @brief Function to specify the optimization problem for 3D COD
%running, with measured IMU data tracking
%>
%> @param   model               Gait3d: Model which should be used for the simulation
%> @param   resultfile          String: Name of the resultfile including path
%> @param   trackingData        TrackingData: Tracking Data containing acceleration and angular velocity data 
%> @param   targetSpeed         Double: Target speed of the movement in x direction in m/s. 
%>                              This target speed will be enforced.
%> @param   isSymmetric         Bool: Specifies wether we assume symmetry of the
%>                              movement. If we assume symmetry, we simulate only one half 
%>                              of gait cycle. This has to fit to the tracking data.
%> @param   initialGuess        String: Filename with path specifying the initial guess 
%> @param   pelvisObjBool       Boolean: Boolean specifying if pelvic rotation objective 
%                               is included in the optimal control problem
%> @param   zDirVelocityObjBool Boolean: Boolean specifying if z-direction
%                               velocity objective is included in the optimal control problem
%> @retval  problem             Collocation: Optimization problem for 2D running
% ======================================================================
function problem = CODrunning3DwthMeasuredIMU(model, resultfile, trackingData, initialGuess, pelvisObjBool, zDirVelocityObjBool, vaeParams)
    %old function code based on running2D: problem = CODrunning3DwthMeasuredIMU(model, resultfile, trackingData, targetSpeed, isSymmetric, initialGuess)
%% Fixed settings
% We can choose the number of collocation nodes.
nNodes = 100; %based on Jiating's code who selected and hardcoded that 
% Most of the time we use backard euler for discretization which is encoded with 'BE'.
Euler = 'BE';
% We usually use the name of the resultfile for the name of the logfile
logfile = resultfile;
% We want to plot intermediate results during solving of the problem.
plotLog = 1;

%% VAE settings
useVAE = false;
if nargin >= 7 && ~isempty(vaeParams)
    useVAE = true;

    % Check VAE parameters
    requiredFields = {'modelPath', 'pythonPath', 'numDofs', 'latentDim'};
    for i = 1:length(requiredFields)
        if ~isfield(vaeParams, requiredFields{i})
            error('Missing required VAE parameter: %s', requiredFields{i});
        end
    end

    % Set default value
    if ~isfield(vaeParams, 'scalerPath')
        vaeParams.scalerPath = '';
    end
    if ~isfield(vaeParams, 'hiddenDim')
        vaeParams.hiddenDim = 512;
    end
    if ~isfield(vaeParams, 'device')
        vaeParams.device = 'cpu';
    end
    if ~isfield(vaeParams, 'weight')
        vaeParams.weight = 10.0;
    end
else
    vaeParams = [];
end

%% Create collocation problem
problem = Collocation(model, nNodes, Euler, logfile, plotLog);


%% Add states and controls including their bounds and initial values to the problem
% General bound values to be used for selected states
p_global_x = [-5, 5];
p_global_y = [-1, 2];
p_global_z = [-5, 5];

% Get upper and lower bounds of the model and resize it
states_min = repmat(model.states.xmin,1,nNodes);
states_max = repmat(model.states.xmax,1,nNodes); % ask Eva when we use N+1

% Get the index of specific states of the model
idxPelvisX = model.extractState('q', 'pelvis_tx');
idxPelvisY = model.extractState('q', 'pelvis_ty');
idxPelvisZ = model.extractState('q', 'pelvis_tz');
idxPelvisRot = model.extractState('q', 'pelvis_rotation'); 
idxPelvVelZ = model.extractState('qdot', 'pelvis_tz');
idxXc = model.extractState('xc'); %'c' in xc/yc/zc stands for contact. 
idxYc = model.extractState('yc'); % these state types (xc,yc,zc) are the global positions of each contact point (in each direction) 
idxZc = model.extractState('zc');

 % Using the extracted states' indices, set their respective min and
 % max bounds, for all nodes, using the global bounds specified
 % earlier
states_min(idxPelvisX, :) = p_global_x(1);
states_max(idxPelvisX, :) = p_global_x(2);
states_min(idxPelvisY, :) = p_global_y(1);
states_max(idxPelvisY, :) = p_global_y(2);
states_min(idxPelvisZ, :) = p_global_z(1);
states_max(idxPelvisZ, :) = p_global_z(2);
states_min(idxXc, :) = p_global_x(1);
states_max(idxXc, :) = p_global_x(2);
states_min(idxYc, :) = p_global_y(1);
states_max(idxYc, :) = p_global_y(2);
states_min(idxZc, :) = p_global_z(1);
states_max(idxZc, :) = p_global_z(2);
% changing the min and max bounds for the allowed pelvic rotation, for
% all time nodes, so that cutting movements that require the pelvis 
% to rotate beyond 90 degrees (the original min and max bounds) 
% are allowed/feasible. Currently setting the bounds to +/-145degrees
% = +/-2.5307 radians
states_min(idxPelvisRot,:) = 0; %LC:latest change -2.5307;
states_max(idxPelvisRot,:) = 2.5307;

% Set the bounds for the first node of some states
states_min(idxPelvisX,1) = 0;
states_max(idxPelvisX,1) = 0;
states_min(idxPelvisZ,1) = 0;
states_max(idxPelvisZ,1) = 0;

if pelvisObjBool == true
    states_min(idxPelvisRot,1) = 0; %comment-out for simulations that dont use the pelvis rotation objective and dont need a 0 initial pelvic rotation
    states_max(idxPelvisRot,1) = 0; %commment-out for simulations that dont use the pelvis rotation objective and dont need a 0 initial pelvic rotation
end

%Constrain initial node of z-direction velocity to 0, for simulations using
%the z-direction velocity objective
if zDirVelocityObjBool == true
    states_min(idxPelvVelZ,1) = 0; %use for simCa
    states_max(idxPelvVelZ,1) = 0; %use for simCa
end

% Add states
problem.addOptimVar('states', states_min, states_max);

% Add controls to the problem using the default bounds 
problem.addOptimVar('controls',repmat(model.controls.xmin,1,nNodes), repmat(model.controls.xmax,1,nNodes));

% Add duration of the movement 
duration = trackingData.variables{49,2}{1,1}(end,1);
problem.addOptimVar('dur',duration, duration);

% Add speed in x direction of the movement. We choose here targetspeed for
% the lower and upper bound to ensure that we have exactly this speed. You
% could also choose other bounds.
%problem.addOptimVar('speed',targetSpeed, targetSpeed);

%% Add the initial guess of the optimization
% After adding all the components to X, we can use a previous solution as
% initial guess. In this case, we use the standing solution we produced
% before.
problem.makeinitialguess(initialGuess); 

%% Add objective terms to the problem
% Set the weights for each objective term (values copied from
% Jiating's code
regularizationWeight = 1e-07; %weight for regularization term. This term ensures realistic and stable solutions (no jerky movements or unrealistic muscle efforts that may violate physiological limits). It typically penlizes the magnitude of control inputs (like u=excitation) to prevent excessive use of certain muscles/actuators 
effortMusclesWeight = 0.7; %weight for effort term of muscles. This term aims at minimize energy expenditure or fatigue. It typically penalizes high muscle activations or force generation. It encourages the system to find solutions optimizing muscle efficiency.
effortTorquesWeight = 0.14; %weight for effort term of torques. This term aims at minimizing mechanical work or energy needed to produce torques at joints. It minimizes high torques. It discourages the system from relying on large torques which might be impractical or inefficent in real-world scenarios. It can reduce the mechanical strain on joints.
trackingWeight = 0.001; %weight for tracking objective. In this simulation we want to track accelerations (accelerometer) and angular velocities (gyroscope)
ratioGyroToAcc = 1000; %the ratio between the accelerometer and gyroscope weights
accWeight = 1; %weight for acceleration 
gyroWeight = accWeight*ratioGyroToAcc; %weight for angular velocities
pelvisRotationWeight = 1; %starting weight= 1


% We add a small regulatization term. This ensures smooth
% movements and helps the solver to converge more easily. 
problem.addObjective(@regTerm, regularizationWeight);

% Add the effort term for muscles.
weightsType = 'volumeweighted'; % In the effort term, we can use 'equal' or 'volumeweighted'. The first uses equal weighting for all muscles independent of their volume. The second weighs them dependent on their volume
exponentEffMusc = 3; % In the effort term, we want to use the cubic neural excitation of the muscles.
speedWeighting = 0;

problem.addObjective(@effortTermMuscles, effortMusclesWeight, weightsType, exponentEffMusc, speedWeighting);

% Add the effort term for torques
exponentEffTorq = 2;
speedWeighting = 0;
problem.addObjective(@effortTermTorques, effortTorquesWeight, exponentEffTorq, speedWeighting);

% Setup and add the tracking term. For this simulation we will be
% tracking accelerations and angular velocities from measured IMU data
trackingDataAcc  = trackingData.extractData('acc');
trackingDataAcc.resampleData(nNodes);
trackingDataAcc.trimData(1, nNodes-1);      
trackingDataGyro = trackingData.extractData('gyro');
trackingDataGyro.resampleData(nNodes); 

%%MEA: Commented-out tracking acceleration to see if only gyro would lead
%%to better results, given that higher speeds affect acceleration accuracy
%%more than they affect gyro
problem.addObjective(@trackAcc, trackingWeight*accWeight, trackingDataAcc);
problem.addObjective(@trackGyro, trackingWeight*gyroWeight, trackingDataGyro);

% Add objective term that requires a specific pelvic rotation to be
% achieved by the final node of the simulation
if pelvisObjBool == true
    problem.addObjective(@pelvisRotationObjectiveForCOD, pelvisRotationWeight);
end

% Add objective term that requires a specific horizontal velocity direction to be
% achieved by the final node of the simulation
if zDirVelocityObjBool == true
    problem.addObjective(@velocityDirectionObjectiveForCOD, pelvisRotationWeight);
end

% VAE objective
if useVAE
    try
        problem.addObjective(@vaeReconstructionTerm, vaeParams.weight, vaeParams);
        fprintf('Successfully added VAE objective!\n')
    catch ME
        error('Failed to add VAE objective: %s', ME.message);
    end
end

%% Add the constaints for keeping dynamic equillibium
problem.addConstraint(@dynamicsFirstNodeConstraint,model.constraints.fmin,model.constraints.fmax)
problem.addConstraint(@dynamicConstraints,repmat(model.constraints.fmin,1,nNodes-1),repmat(model.constraints.fmax,1,nNodes-1));

end