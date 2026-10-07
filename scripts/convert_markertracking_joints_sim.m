function convert_markertracking_joints_sim(srcDir, dstDir)
% convert_markertracking_joints_sim FK-project every sim trial's q(t) onto
% anatomical joint centers, for the per-marker... per-JOINT MPJPE row added
% to markertracking_exploration.ipynb's catplot.
%
% Usage (workstation only -- see convert_markertracking_to_plain.m's
% docstring for why):
% LM_PROJECT=iwse matlab -batch "addpath(genpath('BioMAC-Sim-Toolbox/src')); addpath('scripts'); convert_markertracking_joints_sim('BioMAC-Sim-Toolbox/results/MarkerTracking3D/Participant_02','results_sim/MarkerTracking3D/Participant_02_converted_joints')"
%
% Writes a SEPARATE converted tree from convert_markertracking_to_plain.m
% (same source files, same directory layout) rather than adding fields to
% its output, so the existing angle-RMSE/GRF-RMSE loader path and its
% skip-if-exists files are untouched.
%
% Collocation.extractData's default `settings.angle` excludes pelvis
% orientation (and this model has no wrist DOFs at all -- confirmed via
% Gait3d.nDofs=33 on the workstation, vs the 37 <Coordinate>s in the .osim),
% so a complete q(t) is pulled straight from the raw state instead:
% `result.X(result.problem.idx.states)` is the full nStates x nNodes state
% matrix (same expression Gait3d.showStick/showMarker use), and q is always
% its first `model.nDofs` rows -- verified in radians/meters already (a
% standing pose's pelvis_ty came out ~0.97 m, hip/knee/ankle FK heights were
% anatomically sane) with no unit conversion needed, unlike extractData's
% table which converts angles to degrees for readability.

if nargin < 1 || isempty(srcDir)
    srcDir = 'BioMAC-Sim-Toolbox/results/MarkerTracking3D/Participant_02';
end
if nargin < 2 || isempty(dstDir)
    dstDir = 'results_sim/MarkerTracking3D/Participant_02_converted_joints';
end
srcDir = char(srcDir);
dstDir = char(dstDir);

if ~exist(srcDir, 'dir')
    error('Source directory does not exist: %s', srcDir);
end
if ~exist(dstDir, 'dir')
    mkdir(dstDir);
end

matFiles = list_mat_files(srcDir);
fprintf('Found %d .mat files under %s\n', numel(matFiles), srcDir);
count = 0;
for i = 1:numel(matFiles)
    inFile = matFiles{i};
    outFile = strrep(inFile, [srcDir filesep], [dstDir filesep]);
    outDir = fileparts(outFile);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    if exist(outFile, 'file')
        fprintf('Output file already exists, skipping: %s\n', outFile);
        continue;
    end

    try
        data = load(inFile);
        result = data.result;
        model = result.problem.model;

        jointVarTable = markertracking_joint_variable_table(model);
        jointNames = unique(jointVarTable.name, 'stable');
        nJ = numel(jointNames);

        x = result.X(result.problem.idx.states);
        q = x(1:model.nDofs, :);
        nNodes = size(q, 2);

        jointPos = nan(nNodes, nJ, 3);
        for iNode = 1:nNodes
            m = model.simuMarker(jointVarTable, q(:, iNode));
            jointPos(iNode, :, :) = reshape(m, 3, nJ)';
        end

        S = struct();
        S.converged = result.converged;
        S.dofNames = model.dofs.Properties.RowNames;
        S.q = q;
        S.jointNames = jointNames;
        S.jointPos = jointPos; % nNodes x nJoints x 3 (x,y,z), meters, global frame
    catch e
        fprintf('Failed to convert %s: %s\n', inFile, e.message);
        continue;
    end

    save(outFile, '-struct', 'S', '-v7');
    fprintf('Saved converted file: %s\n', outFile);
    count = count + 1;
end

fprintf('Conversion complete -- %d files written to %s\n', count, dstDir);
end


function files = list_mat_files(root)
    d = dir(fullfile(root, '**', '*.mat'));
    files = fullfile({d.folder}, {d.name});
end
