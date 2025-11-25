%======================================================================
%> @file FAME\MyImplementationFC\scriptCODfromMeasIMU.m
%> @brief Script to start working with old Innsbruck IMU data
%> @example
%> @details
%> This is a script which you can use if you want to run a IMU-tracking
%simulation for one trial of one participant based on the the old
%Innsbruck data (the ones with baseline and followup measurements
%before and after the training programs)
%>
%> @author Maria Eleni Athanasiadou
%> @date October, 2024
%======================================================================

% ======================================================================
%> @brief Main function to run an IMU-tracking simulation, using the old
%> Innsbruck data, using the opensim models that Jiating had created
%>
%> @param   path        Should be the path to the BioMAC-Sim-Toolbox folder.
%>                      Added this param to adjust the script for use with the HPC
%> @param   pID         pID=participant ID. Format: 's01', 's21', 's23'
%> @param   trialType   Can be 'Baseline' or 'Followup'
%> @param   settings    Defines which objectives and what kind of initial
%>                      guess will be used for the optimal control problem.
%>                      Options: 'ffs', 'tfs', 'fts' and 'ftc'. See code
%>                      for more details
% ======================================================================

function scriptCODfromMeasIMU(path, pID, trialType, settingsInpt) 


    
    
    %% Settings
    % Get path of this script
    %filePath = fileparts(mfilename('fullpath'));
    % Path to your repository
    %path2repo = [filePath filesep '..' filesep '..' filesep];
    path2repo = path;
    addpath(genpath(path2repo));
    
    % Fixed settings
    %participant.ID = 's01';
    participant.ID = pID;
    %participant.trialType = 'Followup'; %can be Baseline or Followup
    participant.trialType = trialType;
    abbrevTrial = trialType(1:1);
    trialTypeLowercase = lower(participant.trialType);
    dataFolder     = 'data/MaroCODsims/DataStructsForInnsbruckOldDtaFromJiating';    % Relative from the path of the repository
    participant.modelFile = [path2repo dataFolder '/model/' ...
        'gait3d_pelvis213_Innsbruck_scaled_' participant.ID '_' ...
        trialTypeLowercase '.osim'];
    %DEL: participant.dataFile       = 'data_Erlangen_cut135_s01_3D_1_measuredIMU.mat'; % datastruct object created by Jiating's code
    %modelFile      = 'gait3d_pelvis213.osim';        % Name of our default model with adapted pelvis rotation order in constrast to Hamners model
    resultFolder   = ['results/MaroCODsims' filesep participant.ID filesep participant.trialType filesep]; % Relative from the path of the repository
    
    % VAE Configuration
    %enable flag
    useVAE = true;
    
    if useVAE
        %vaeFilePath = fullfile(path, 'ExampleScripts', 'vaeFile');
        vaeParams.modelPath  = which('BiomechPriorVAE_best.pth');
        vaeParams.scalerPath = which('scaler.pkl');
        %vaeParams.pythonPath = ;

        fprintf('Model Path:   %s\n', vaeParams.modelPath);
        fprintf('Scaler Path:  %s\n', vaeParams.scalerPath);
        
        vaeParams.numDofs = 54; %excluding pelvis
        vaeParams.latentDim = 24;
        vaeParams.hiddenDim = 512;
        vaeParams.device = 'cpu';
        vaeParams.weight = 0.5;
    
        fprintf('VAE constraints enabled\n')
    else
        fprintf('VAE constraints disabled\n')
    end
    
    %% Initalization
    % Get date
    dateString = datestr(date, 'ddmmyyyy');
    
    % Get absolute file names
    resultFileStanding = [path2repo,filesep,resultFolder,participant.ID,abbrevTrial,settingsInpt, '_', dateString,'_', mfilename,'_standing'];
    resultFileCODRunning  = [path2repo,filesep,resultFolder,dateString, '_', settingsInpt, filesep, participant.ID,abbrevTrial,settingsInpt,'_', dateString,'_', mfilename,'_CODrunning'];
    sbjDataDir = [path2repo,filesep,dataFolder,filesep,participant.ID,filesep,participant.trialType, filesep];
    dataFile=dir(strcat([path2repo,filesep,dataFolder,filesep,participant.ID,filesep,participant.trialType,filesep], '*.mat')).name;
    sbjDataFile = [sbjDataDir dataFile];
    %DEL: dataFile           = [path2repo,filesep,dataFolder,filesep,participant.ID,filesep,participant.trialType,filesep,'Processed',filesep,participant.dataFile];
    dataPath           = [path2repo,filesep,dataFolder];
    modelPath          = participant.modelFile;%[path2repo,filesep,dataFolder,filesep,participant.ID,filesep,participant.trialType,filesep,'OpenSimFiles',filesep,participant.modelFile];
    
    addpath(dataPath);
    
    
    % Create resultfolder if it does not exist
    resultFolderPath = [path2repo,filesep,resultFolder];
    if ~exist(resultFolderPath, 'dir')
        mkdir(resultFolderPath);
    end
    
    %% Optimal control problem Settings
    if strcmp(settingsInpt, 'ffs')
        pelvisObjectiveBool = false;
        zDirVelocityObjectiveBool = false; 
        initGuess = 'stand';
    elseif strcmp(settingsInpt, 'tfs')
        pelvisObjectiveBool = true;
        zDirVelocityObjectiveBool = false; 
        initGuess = 'stand';
    elseif strcmp(settingsInpt, 'fts')
        pelvisObjectiveBool = false;
        zDirVelocityObjectiveBool = true; 
        initGuess = 'stand';
    elseif strcmp(settingsInpt, 'ftc')
        pelvisObjectiveBool = false; % can be false or true, to specify if this objective is included in the optimal control problem
        zDirVelocityObjectiveBool = true; % can be false or true, to specify if this objective is included in the optimal control problem
        initGuess = 'cod'; % can be 'stand' or 'cod', to select corresponding initial guess file
    end
    % cod init guess might be preferred when zDirVelocityObjectiveBool = 1,
    % because this objective has struggled with finding a good simulation
    % result in the past. The COD init guess steers it towards a 'better'
    % solution
    
    %% If-statement checks what objectives the following COD simulation will
    %include. If the pelvis rotation and z-dir velocity are false, then this
    %should be the very first simulation to be run and the corresponding folder
    %will be created for it and the path where the result will be stored is 
    %updated accordingly
    if pelvisObjectiveBool == true || zDirVelocityObjectiveBool == true
        % Create result sub-folder for storing specific simulation result,
        % if it does not exist
        [parentPath, ~, ~]=fileparts(resultFileCODRunning);
        if ~exist(parentPath, 'dir')
            mkdir(parentPath); 
        end
    else
        % Create result sub-folder for storing specific simulation result,
        % if it does not exist
        resultFileCODRunning  = [path2repo,filesep,resultFolder,dateString, '-CODinitGuess', filesep, dateString,'_', mfilename,'_CODrunning'];
        [parentPath, ~, ~]=fileparts(resultFileCODRunning);
        if ~exist(parentPath, 'dir')
            mkdir(parentPath); 
        end
    end
    
    %% Standing: Simulate standing with minimal effort without tracking data for one point in time (static)
    %%or if standing simulation exists already, load it
     % For s01 baseline trial:
     % file 2025_02_03_scriptCODfromMeasIMU_standing.mat is the standing file
     % (initial guess) generated with the new toolbox 
     % file 2024_12_18_scriptCODfromMeasIMU_standing.mat is the standing file
     % (initial guess) generated with the old toolbox
     % Depending on which toolbox is used to run simulations, an initial guess
     % made with the respective toolbox should be used.
     % The same holds for the COD initial guess, where
     % 2024_12_18_scriptCODfromMeasIMU_CODrunning.mat=> old toolbox
     % 2025_02_03_scriptCODfromMeasIMU_CODrunning.mat=> new toolbox
    
     % searches for a standing file match, that can be loaded as the initial
     % guess, rather than running a standing simulation. If such a file doesn't
     % exist, it is created using the simulation (else statement)
     standingInitGuess = '*_standing.mat';
     cd([path2repo, filesep, resultFolder]);
     fileMatch = dir(standingInitGuess);
     cd(path2repo);
    
     if ~isempty(fileMatch) 
         resultFileStanding = [fileMatch.folder filesep fileMatch.name];
     else
        % Create an instance of our 3D model class using the default settings
        model = Gait3d(modelPath);
        % => We use a mex function for some functionality of the model. This was
        % automatically initialized with the correct settings for the current
        % model. (see command line output)    

        resultStandingArray = cell(1,5);
        objectiveValues = cell(1,5);
        for i=1:5
            % Call IntroductionExamples.standing3D() to specify the optimizaton problem
            % => Have a look into it ;)
            %Old toolbox: problemStanding = IntroductionExamples.standing3D_pelvis213(model, resultFileStanding);
            
            %new toolbox
            % problemStanding = IntroductionExamples.standing3D(model, resultFileStanding);
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
            solver.setOptionField('max_iter', 10000);
            solver.setOptionField('tol', 0.0001);%0.0000001
            solver.setOptionField('dual_inf_tol', 0.0001);

            solver.setOptionField('constr_viol_tol', 0.001);%0.000001

            % Solve the optimization problem
            resultStandingArray{1,i} = solver.solve(problemStanding);
            objectiveValues{1,i} = resultStandingArray{1,i}.info.objective;
            
        end

        validIdx=false(1, 5);
        convergedResultsArray = cell(1,5);
        convergedObjectiveValues = cell(1,5);

        for ind=1:5
            if resultStandingArray{1,ind}.converged == 1
                validIdx(ind)=1;
                convergedResultsArray{1,ind}=resultStandingArray{1,ind};
                convergedObjectiveValues{1,ind} = resultStandingArray{1,ind}.info.objective;
            end
        end


        if any(validIdx)
            disp('At least one standing simulation has converged and can be used as initial guess');
            [~,iMin] = min([convergedObjectiveValues{:}]);
            resultStanding = resultStandingArray{1,iMin};
            
            % Save the result
            resultStanding.save(resultFileStanding);
            
            % To plot the result we have to extract the states x from the result vector X
            x = resultStanding.X(resultStanding.problem.idx.states);
            
            % Now, we can plot the stick figure visualizing the result
            figure();
            resultStanding.problem.model.showStick(x);
            title('3D Standing');
            
            % If the model is standing on the toes, this is a local optimum. Rerun this
            % section and you will find a different solution, due to a different random
            % initial guess.
        else
            error('No standing simulation has converged. Code execution will stop here.');
        end

       
     end
     %return
     
    % The following -if/else- section was used when only 1 participant and trial
    % type was used for simulations. Since more participants are integrated
    % this section is outdated and has been replaced by the one above
    %  if exist('/home/od80izej/Documents/BioMAC-Sim-Toolbox/results/MaroCODsims/2025_02_03_scriptCODfromMeasIMU_standing.mat', 'file')
    %      resultFileStanding='/home/od80izej/Documents/BioMAC-Sim-Toolbox/results/MaroCODsims/2025_02_03_scriptCODfromMeasIMU_standing.mat';
    %  else
    %     % Create an instance of our 3D model class using the default settings
    %     model = Gait3d(modelPath);
    %     % => We use a mex function for some functionality of the model. This was
    %     % automatically initialized with the correct settings for the current
    %     % model. (see command line output)
    %     
    %     % Call IntroductionExamples.standing3D() to specify the optimizaton problem
    %     % => Have a look into it ;)
    %     %Old toolbox: problemStanding = IntroductionExamples.standing3D_pelvis213(model, resultFileStanding);
    %     
    %     %new toolbox
    %     problemStanding = IntroductionExamples.standing3D(model, resultFileStanding);
    %     
    %     % Create an object of class solver. We use most of the time the IPOPT here.
    %     solver = IPOPT();
    %     
    %     % Change settings of the solver
    %     solver.setOptionField('tol', 0.0000001);
    %     solver.setOptionField('constr_viol_tol', 0.000001);
    %     
    %     % Solve the optimization problem
    %     resultStanding = solver.solve(problemStanding);
    %     
    %     % Save the result
    %     resultStanding.save(resultFileStanding);
    %     
    %     % To plot the result we have to extract the states x from the result vector X
    %     x = resultStanding.X(resultStanding.problem.idx.states);
    %     
    %     % Now, we can plot the stick figure visualizing the result
    %     figure();
    %     resultStanding.problem.model.showStick(x);
    %     title('3D Standing');
    %     
    %     % If the model is standing on the toes, this is a local optimum. Rerun this
    %     % section and you will find a different solution, due to a different random
    %     % initial guess.
    % end
    
    % Make sure to name the folder of the 'basic' simulation (the one without
    % added objectives) as 'xxx-CODinitGuess' in order to be able to find if
    % such a simulation exists and load it as the COD initial guess (which can
    % be used, if needed, for the simulation including the z-direction 
    % velocity objective.
    
    
    % If-statement sets the CODinitialGuess variable, if the initGuess
    % parameter requires a COD as initial guess, if a corresponding
    % CODinitGuess folder exists, if that folder contains any .mat file (it is
    % possible that the folder exists because it was just made and it curently
    % contains not .mat file. But this is erroneous use (wrong initial params
    % set), because you cannot run a sim that needs a COD as init guess and not
    % have run a sim that uses a standing guess as init guess. If that had been
    % done, then the folder would by default be called xxx-CODinitGuess and
    % it would contain a .mat file already) and if 1 or more .mat files are
    % found, it either selects the singular file or requires user input to
    % choose between .mat files.
    if strcmp(initGuess, 'stand')
        warning('Bypassing setting CODinitialGuess variable.');
    else
        searchStr = '-CODinitGuess';
        dirInfo = dir(resultFolderPath);
        dirNames = {dirInfo([dirInfo.isdir]).name};
        matchingFolder = dirNames(contains(dirNames,searchStr));
        
        if ~isempty(matchingFolder)
            if length(matchingFolder)>1
                warning('More than two COD simulations that are suitable as initial guesses have been found.');
                % TODO check if any of the COD init guess folder contains a .mat file. If
                % no one contains a .mat file set CODinitialGuess=''; and this will throw an error down the line
                % If only one contains a .mat file, set that as COD init guess. If
                % all folder options contain .mat file, prompt user to choose
                matfileCntr=0;
                matFilesArray={}; %preallocates space for a max of 5 .mat COD files that can be used as init guess
                % in reality we shouldn't get more than 1 or 2.
    
                for i=1:length(matchingFolder)
                    subfolderPath=fullfile(resultFolderPath, matchingFolder{i});
                    matFiles=dir(fullfile(subfolderPath, '*.mat'));
    
                    if ~isempty(matFiles)
                        matfileCntr=matfileCntr+1;
                        matFilesArray{end+1}=[matFiles.folder filesep matFiles.name]; 
                    end
                end
    
                if matfileCntr==0
                    CODinitialGuess='';
                elseif matfileCntr==1
                    CODinitialGuess=matFilesArray{1};
                elseif matfileCntr>1
                    fprintf('Available choices:\n');
                    for i=1:length(matFilesArray)
                        fprintf('%d: %s\n', i, matFilesArray{i});
                    end
                    while true
                        userInput = input(sprintf('Select a number (1-%d) to choose one of the above files as the COD initial guess:', length(matFilesArray)));
                        if isnumeric(userInput) && isscalar(userInput) && userInput>=1 && userInput<= length(matFilesArray)
                            break;
                        else
                            fprintf('Invalid choice. Please enter a number between 1 and %d.\n', length(matFilesArray));
                        end
                    end
                    CODinitialGuess=matFilesArray{userInput};
                end
            else
                codInitGuessSearchStr = '*.mat';
                cd([resultFolder matchingFolder{1}]);
                fileMatchCOD = dir(codInitGuessSearchStr);
                CODinitialGuess=fileMatchCOD.name;
                cd(path2repo);
            end
        else
            CODinitialGuess='';
        end
    end
    
    % To be Deleted:
    % if exist('/home/od80izej/Documents/BioMAC-Sim-Toolbox/results/MaroCODsims/10_03_2025/2025_03_10_scriptCODfromMeasIMU_CODrunning.mat', 'file')
    %     CODinitialGuess='/home/od80izej/Documents/BioMAC-Sim-Toolbox/results/MaroCODsims/10_03_2025/2025_03_10_scriptCODfromMeasIMU_CODrunning.mat';
    % end
    
    
    
    %% Running: Simulate COD running with minimal effort while tracking IMU data
    % Load tracking data struct and create a TrackingData object
    trackingData = TrackingData.loadStruct(sbjDataFile);
    
    %% Q: is a targetSpeed needed??
    % % Set a specifc targetspeed which we want to ensure. The target speed has
    % % to fit the trackingData and the use of it is optional. 
    % targetSpeed = 3.5; % m/s
    
    % Create and automatically initalize an instane of our 3D model class.
    % TODO: consider scaling model in the future, or using the
    % already-scaled models from Jiating's code (there exists a different
    % scaled model for each participant). Scaling uses the height and mass
    % of the subject.
    % code from script2d: model = Gait3d(modelFile, trackingData.subjectHeight, trackingData.subjectMass);
    model = Gait3d(modelPath);
    
    % The data we use in this example was recorded using a treadmill. This is
    % important since we have the change the coeffienct of the air drag in the
    % model. 
    model.drag_coefficient = 0; %M: keep it 0 for now, since we consider indoor COD running
    
    % => You may have noticed that the mex file was again initialized after
    % changing the air drag. This is necessary to update the model parameters
    % of the mex file. 
    
    % Call IntroductionExamples.running2D() to specify the optimizaton problem
    % => Have a look into it ;)
    %isSymmetric = 1; %M: symmetry not included bc it simulates half a
    %gait cycle which is not usefull for COD running
    
    %Q: For now, will try out using the 3D standing initial guess. Should I
    %not?
    % Set inital guess based on initial parameter
    if strcmp(initGuess, 'stand')
        initialGuess = resultFileStanding; %use this initial guess for simulations
        %with small or big pelvic rotation bounds (90-145 deg), with a pelvic
        %rotation objective (not a velocity direction objective) and a
        %constrained initial pelvic rotation (=0).
    elseif strcmp(initGuess, 'cod')
        initialGuess = CODinitialGuess; %use this initial guess for the simulation 
        % using the velocity direction objective, the big pelvic rotation
        % bounds, the constrained z-direction initial velocity and the uncostrained
        % initial pelvic rotation angle
    end
    
    
    
    %% Set-up problem: objective terms, constraints, etc
    if useVAE
        straightRunningVAE = vaeParams;
        problemRunning = MaroScripts.CODrunning3DwthMeasuredIMU(model, resultFileCODRunning, trackingData, initialGuess, pelvisObjectiveBool, zDirVelocityObjectiveBool, straightRunningVAE);
    else
        problemRunning = MaroScripts.CODrunning3DwthMeasuredIMU(model, resultFileCODRunning, trackingData, initialGuess, pelvisObjectiveBool, zDirVelocityObjectiveBool);
    end
    
    % Create solver and change solver settings
    solver = IPOPT();
    solver.setOptionField('max_iter', 15000);
    solver.setOptionField('tol', 0.0005);
    solver.setOptionField('dual_inf_tol', 0.0005);
    
    % Solve the optimization problem and save the result
    resultRunning = solver.solve(problemRunning);
    resultRunning.save(resultFileCODRunning);
    
    % Now, we can use our report function to look at the results. We can use the
    % default settings for extracting variables and plotting by calling the
    % function without inputs.
    % However, we additionally want to plot the initial guess which was used
    % and adapt the figure size.
    settings.plotInitialGuess = 1;
    style.figureSize = [0 0 16 26];
    % When also giving a filename as input, the function will automatically
    % create a pdf summarizing all information and plots. Have a look at it!
    resultRunning.report(settings, style, resultFileCODRunning);
    
    % If you do not want to plot the variables, but want to only extract specific
    % biomechanical variables, you can instead use the extractData function.
    % To obtain the muscle forces of all muscles additionally to the default
    % variables, we can adapt the settings struct.
    settings.muscleForce = model.muscles.Properties.RowNames;
    % For symmetric simulations, we can specify whether the returned
    % simVarTable should reflect a full gait cycle or only the simulated
    % motion, i.e. half a gait cycle.
    getFullCycle = 1;
    simVarTable = resultRunning.problem.extractData(resultRunning.X, settings, [], getFullCycle);
    % Open now simVarTable and inspect it. You will see that is has a similar
    % structure as the table in trackingData.variables.
    
    % If you have reference data in the tracking data format, you can use the
    % input argument variableTable of extractData():
    % variableTable = referenceData.variables;
    % This data will be plotted as "measured" data in the report function and
    % saved as mean_extra and var_extra in the simVarTable.
    
    % It is worth saving this simVarTable in a sparate file to reuse when
    % performing the evaluation of your study. If you also added the reference
    % data before calling the report, you might have already everything you
    % need for evaluation in one file.
    
    % If you generated multiple simulations, you can use Collocation.plotMultSimVarTables()
    % to plot multiple results into one graph and Collocation.plotMeanSimVarTables()
    % to plot the mean and variance of multiple results into one graph.
    
    % We can also write a movie of the movement
    resultRunning.problem.writeMovie(resultRunning.X, resultRunning.filename);
    
    % We can also compute the metabolic cost in J/m/kg which is need for the movement.
    % This is only the metabolic cost of half of the gait cycle which was
    % simulated. 
    resultRunning.problem.getMetabolicCost(resultRunning.X);
    
    osimMotionPath = [parentPath filesep participant.ID abbrevTrial settingsInpt '_' dateString '_OsimFiles'];
    if ~exist(osimMotionPath, 'dir')
        mkdir(osimMotionPath);
        cd(osimMotionPath);
    else
        cd(osimMotionPath);
    end
    resultRunning.problem.writeMotionToOsim(resultRunning.X, [participant.ID,abbrevTrial,settingsInpt,'_',dateString, '_Osim']);


end