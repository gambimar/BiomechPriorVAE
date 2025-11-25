%======================================================================
%> @file IntroductionExamples/script3D.m
%> @brief Script to start working with the Bio-Sim-Toolbox for 3D models
%> @example
%> @details
%> This is a script which you can use if you work with our code for the
%> first time. The settings are based on the following paper:
%> Nitschke, M., Dorschky, E., Heinrich, D., Schlarb, H., Eskofier, B. M.,
%> Koelewijn, A. D., & van den Bogert, A. J. (2020). Efficient trajectory
%> optimization for curved running using a 3D musculoskeletal model with
%> implicit dynamics. Scientific reports, 10(1), 17655.
%>
%> @author Marlies Nitschke, Anne Koelewijn
%> @date September, 2024
%======================================================================


clear all 
close all
clc

%% Settings
% Get path of this script
filePath = fileparts(mfilename('fullpath'));
% Path to your repository
path2repo = [filePath filesep '..' filesep '..' filesep];


% Fixed settings
dataFolder     = 'data/IntroductionExamples';    % Relative from the path of the repository
dataFile       = 'running_3D.mat';               % Straight running data used in 2020 paper
dataFileCurved = 'curvedRunning_3D.mat';         % Curved running data used in 2020 paper
modelFile      = 'gait3d_pelvis213.osim';        % Name of our default model. You can add a new model in the osim_files folder of Gait3d, but you can also specify the path to a model similar to how it is done for the dataFile
resultFolder   = 'results/IntroductionExamples'; % Relative from the path of the repository

% VAE Configuration
%enable flag
useVAE = true;

% 27: q, 31: q+F, 54: q+qdot, 58: q+qdot+F, 56: q+qdot+M_ankle
nDimVae = 81;
if nDimVae > 50
    latent_size = 24;
else
    latent_size = 24;
end

if useVAE
    vaeParams.modelPath = which(strcat('BiomechPriorVAE_best_',string(nDimVae),'.pth'));
    vaeParams.scalerPath = which(strcat('scaler_',string(nDimVae),'.pkl'));
    vaeParams.pythonPath = filePath;
    
    vaeParams.numDofs = nDimVae; %excluding pelvis
    vaeParams.latentDim = latent_size;
    vaeParams.hiddenDim = 512;
    vaeParams.device = 'cpu';
    vaeParams.weight = 1e3;

    fprintf('VAE objective enabled\n')
else
    fprintf('VAE objective disabled\n')
end


%% Initalization
% Get date
dateString = datestr(date, 'yyyy_mm_dd');

% Get absolute file names
resultFileStanding      = [path2repo,filesep,resultFolder,filesep,dateString,'_', mfilename,'_standing'];
resultFileRunning       = [path2repo,filesep,resultFolder,filesep,dateString,'_', mfilename,'_running'];
resultFileCurvedRunning = [path2repo,filesep,resultFolder,filesep,dateString,'_', mfilename,'_curvedRunning'];
dataFile                = which(dataFile);
dataFileCurved          = [path2repo,filesep,dataFolder,  filesep,dataFileCurved];

% Create resultfolder if it does not exist
if ~exist([path2repo,filesep,resultFolder], 'dir')
    mkdir([path2repo,filesep,resultFolder]);
end


%% Standing: Simulate standing with minimal effort without tracking data for one point in time (static)
% Create an instane of our 3D model class using the default settings
model = Gait3d(modelFile);
% => We use a mex function for some functionality of the model. This was
% automatically initialized with the correct settings for the current
% model. (see command line output)

if useVAE
    standingVAE = vaeParams;
    % Call IntroductionExamples.standing3D() to specify the optimizaton problem
    % => Have a look into it ;)
    problemStanding = standing3D(model, resultFileStanding, standingVAE);
else
    problemStanding = standing3D(model, resultFileStanding);
end

% Create an object of class solver. We use most of the time the IPOPT here.
solver = IPOPT();

% Change settings of the solver
solver.setOptionField('tol', 1e-4); % 0.0000001
solver.setOptionField('constr_viol_tol', 1e-3); % 0.000001
solver.setOptionField('dual_inf_tol', 1e-4);
solver.setOptionField('acceptable_tol', 1e-4); 
%solver.setOptionField('max_iter',2)

% Solve the optimization problem
resultStanding = solver.solve(problemStanding);

% Save the result
resultStanding.save(resultFileStanding);

% To plot the result we have to extract the states x from the result vector X
x = resultStanding.X(resultStanding.problem.idx.states);

% Now, we can plot the stick figure visualizing the result
standingFig = figure();
resultStanding.problem.model.showStick(x);
title('3D Standing'); 
current_t = datetime('now', 'Format', 'yyyyMMdd_HHmmss');
current_t_str = char(current_t);
if useVAE
    filenameFig = ['standing_VAE_' current_t_str '.png'];
else
    filenameFig = ['standing_' current_t_str '.png'];
end

try
    exportgraphics(standingFig, filenameFig);
catch
    saveas(standingFig, filenameFig);
end

% If the model is standing on the toes, this is a local optimum. Rerun this
% section and you will find a different solution, due to a different random
% initial guess.

%% Straight running: simulate running with minimal effort while tracking data
% Simulation settings.
N = 50; % number of collocation nodes
sym = 1; % do not simulate symmetric movement
W.effMuscles = 0; % Weight of effort term in objective 1000
W.effMusclesAct = 1e3; % According to Nitschke, 2019
W.effMusclesTor = 1; %1.0e3 / 10;
W.cot = 0; % 5.0e2 / 92 / 74.9646;
W.effTorques = 0;    % Weight of torque term in objective
W.reg        = 0; % Weight of regularization term in objective
W.track      = 0;    % Weight of tracking term in objective
W.dur        = 0;    % No predefined duration
W.speed      = Inf;  % Predefined speed
initialGuess = resultFileStanding;

% Load and resample tracking data 
trackingData = TrackingData.loadStruct(dataFile);
trackingData.preprocessData(N);
% Extract speed and duration
targetspeed_x =  trackingData.variables.mean{strcmp(trackingData.variables.type,'speed') & strcmp(trackingData.variables.name,'x')}; % speed in x direction
targetspeed_z =  trackingData.variables.mean{strcmp(trackingData.variables.type,'speed') & strcmp(trackingData.variables.name,'z')}; % speed in z direction
targetdur =  trackingData.variables.mean{strcmp(trackingData.variables.type,'duration')}; % duration

fprintf('targetspeed_x: %f', targetspeed_x);
fprintf('targetspeed_z: %f', targetspeed_z);
fprintf('targetdur: %f', targetdur);

% Create and automatically initalize an instance of our 2D model class.  
% To fit the tracking data we have to scale the default model. This is done
% using the height and mass of the participant.
model = Gait3d(modelFile);

% Call IntroductionExamples.running3D() to specify the optimizaton problem
% => Take a look inside the function ;)
if useVAE
    straightRunningVAE = vaeParams;
    % Call IntroductionExamples.standing3D() to specify the optimizaton problem
    % => Have a look into it ;)
    problemRunning = running3D(model,trackingData,initialGuess,resultFileRunning,N,sym,W, targetspeed_x, targetspeed_z, targetdur, straightRunningVAE);
else
    problemRunning = running3D(model,trackingData,initialGuess,resultFileRunning,N,sym,W, targetspeed_x, targetspeed_z, targetdur);
end

%problemRunning.derivativetest
% Create solver and change solver settings
solver = IPOPT();
solver.setOptionField('max_iter', 4500);
solver.setOptionField('tol', 5e-4); % 0.0005
solver.setOptionField('print_level',5);
%solver.setOptionField('dual_inf_tol', 1e-6);
%solver.setOptionField('constr_viol_tol',1e-6);
%solver.setOptionField('acceptable_tol', 1e-6); 

% Solve the optimization problem and save the result. 
resultRunning = solver.solve(problemRunning);
resultRunning.save(resultFileRunning); 


%% Result extraction
idxStandingJointsAllNodes = problemStanding.idx.states(problemStanding.model.extractState('q'), 1:problemStanding.nNodes);
for nodeIdx = 1:problemStanding.nNodes
    standingJoints(:, nodeIdx) = resultStanding.X(idxStandingJointsAllNodes(:, nodeIdx));
end

idxRunningJointsAllNodes = problemRunning.idx.states(problemRunning.model.extractState('q'), 1:problemRunning.nNodes);
for nodeIdx = 1:problemRunning.nNodes
    runningJoints(:, nodeIdx) = resultRunning.X(idxRunningJointsAllNodes(:, nodeIdx));
end

nDimVaeStr = int2str(nDimVae);
if useVAE
    filenameStanding = ['standingJoints_VAE_' current_t_str '_' nDimVaeStr '.mat'];
    filenameRunning = ['runningJoints_VAE_' current_t_str '_' nDimVaeStr '.mat'];
    filenameCurvedRunning = ['curvedRunningJoints_VAE_' current_t_str '_' nDimVaeStr '.mat'];
    filenameResult = ['running_VAE_q_qdot_' current_t_str '_' nDimVaeStr '.mat'];
else
    filenameStanding = ['standingJoints_' current_t_str '.mat'];
    filenameRunning = ['runningJoints_' current_t_str '.mat'];
    filenameCurvedRunning = ['curvedRunningJoints_' current_t_str '.msat'];
    filenameResult = ['running.mat'];
end

save(filenameStanding, 'standingJoints');
save(filenameRunning, 'runningJoints');

save(filenameResult, "resultRunning");

%% Plot
figure;
subplot(3,1,1)
plot([runningJoints(7,:) runningJoints(14,:)].*180/pi)
title('Hip Flexion Angle')

subplot(3,1,2)
plot([runningJoints(10,:) runningJoints(17,:)].*180/pi)
title('Knee Flexion Angle')

subplot(3,1,3)
plot([runningJoints(11,:) runningJoints(18,:)].*180/pi)
title('Ankle Flexion Angle')