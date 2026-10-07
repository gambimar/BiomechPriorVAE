function convert_markertracking_reference_joints(refDir, modelFile, dstDir)
% convert_markertracking_reference_joints FK-project the reference (IK-derived)
% q(t) from convert_markertracking_reference.m's `inverse` onto the same
% anatomical joint centers as convert_markertracking_joints_sim.m, using the
% SAME Gait3d model/FK call -- this is the joint-center ground truth for the
% MPJPE row (there is no genuine mocap ground truth for joint CENTERS, only
% markers, so the reference has to go through FK too, on the IK solution).
%
% Usage (workstation only -- see convert_markertracking_to_plain.m):
% LM_PROJECT=iwse matlab -batch "addpath(genpath('BioMAC-Sim-Toolbox/src')); addpath('scripts'); convert_markertracking_reference_joints('results_sim/MarkerTracking3D/Participant_02/reference','BioMAC-Sim-Toolbox/data/MarkerTracking/Participant_02/Participant_02.osim','results_sim/MarkerTracking3D/Participant_02/reference_joints')"
%
% Reads the already-converted `reference/*.mat` files (convert_markertracking_
% reference.m's output, a plain table2struct dump with no settings-filtering,
% so it has the FULL <Coordinate> set from the .osim -- confirmed on the
% workstation: inverse.variables carries all 3 pelvis-orientation angle rows
% plus every other angle/translation coordinate). Only the `model.dofs`
% subset (33 of those 37 -- this Gait3d model has no wrist DOFs at all) is
% used to build q; angle rows are deg, translation rows are mm (both
% confirmed against convert_markertracking_to_plain.m's sim output, which is
% deg/m -- reference translations need an extra /1000 sim's don't).

if nargin < 1 || isempty(refDir)
    refDir = 'results_sim/MarkerTracking3D/Participant_02/reference';
end
if nargin < 2 || isempty(modelFile)
    modelFile = 'BioMAC-Sim-Toolbox/data/MarkerTracking/Participant_02/Participant_02.osim';
end
if nargin < 3 || isempty(dstDir)
    dstDir = 'results_sim/MarkerTracking3D/Participant_02/reference_joints';
end
refDir = char(refDir);
modelFile = char(modelFile);
dstDir = char(dstDir);

if ~exist(dstDir, 'dir')
    mkdir(dstDir);
end

model = Gait3d(modelFile);
jointVarTable = markertracking_joint_variable_table(model);
jointNames = unique(jointVarTable.name, 'stable');
nJ = numel(jointNames);
dofNames = model.dofs.Properties.RowNames;
nDofs = model.nDofs;

d = dir(fullfile(refDir, '*.mat'));
fprintf('Found %d reference .mat files under %s\n', numel(d), refDir);
count = 0;
for i = 1:numel(d)
    inFile = fullfile(d(i).folder, d(i).name);
    outFile = fullfile(dstDir, d(i).name);
    if exist(outFile, 'file')
        fprintf('Output file already exists, skipping: %s\n', outFile);
        continue;
    end

    try
        data = load(inFile);
        inv = data.inverse.variables;

        firstDofIdx = find(strcmp(inv.name, dofNames{1}), 1);
        nSamples = numel(inv.mean{firstDofIdx});
        q = nan(nDofs, nSamples);
        for iDof = 1:nDofs
            idx = find(strcmp(inv.name, dofNames{iDof}), 1);
            if isempty(idx)
                error('DOF %s not found in inverse.variables', dofNames{iDof});
            end
            vec = inv.mean{idx}(:)';
            switch inv.type{idx}
                case 'angle'
                    vec = deg2rad(vec);
                case 'translation'
                    vec = vec / 1000; % mm -> m
                otherwise
                    error('Unexpected type ''%s'' for DOF %s', inv.type{idx}, dofNames{iDof});
            end
            q(iDof, :) = vec;
        end

        jointPos = nan(nSamples, nJ, 3);
        for iSample = 1:nSamples
            m = model.simuMarker(jointVarTable, q(:, iSample));
            jointPos(iSample, :, :) = reshape(m, 3, nJ)';
        end

        S = struct();
        S.movement = data.movement;
        S.trial = data.trial;
        S.dofNames = dofNames;
        S.q = q;
        S.jointNames = jointNames;
        S.jointPos = jointPos; % nSamples x nJoints x 3 (x,y,z), meters, global frame
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
