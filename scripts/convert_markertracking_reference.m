function convert_markertracking_reference(srcDir, dstDir)
% convert_markertracking_reference Convert measured+inverse MarkerTracking3D reference data to plain MAT
%
% Usage (from terminal, run on the workstation -- see convert_markertracking_to_plain.m):
% LM_PROJECT=iwse matlab -batch "addpath(genpath('BioMAC-Sim-Toolbox/src')); convert_markertracking_reference('BioMAC-Sim-Toolbox/data/MarkerTracking/Participant_02','results_sim/MarkerTracking3D/Participant_02/reference')"
%
% Each raw *_DataStructMeasured.mat / *_DataStructInverse.mat holds a
% top-level `dataStruct` struct whose `variables` field is itself a MATLAB
% table (MCOS), unreadable by plain scipy -- same problem as the Result
% objects in convert_markertracking_to_plain.m, different cause. Measured is
% the raw marker/GRF recording; Inverse is the same trial's IK/ID-derived
% angles/moments/translations/markers computed from the FULL dense marker
% set, independent of any of our sparse-tracking sims -- exactly the
% held-out-channel ground truth Experiment 3 needs. This does not vary by
% sparsity condition, so unlike convert_markertracking_to_plain.m there is
% one output file per movement (not per condition), holding both measured
% and inverse under top-level `measured`/`inverse` fields.

if nargin < 1 || isempty(srcDir)
    srcDir = 'BioMAC-Sim-Toolbox/data/MarkerTracking/Participant_02';
end
if nargin < 2 || isempty(dstDir)
    dstDir = 'results_sim/MarkerTracking3D/Participant_02/reference';
end
srcDir = char(srcDir);
dstDir = char(dstDir);

if ~exist(srcDir, 'dir')
    error('Source directory does not exist: %s', srcDir);
end
if ~exist(dstDir, 'dir')
    mkdir(dstDir);
end

% (movement folder under srcDir, trial id, output movement name)
movements = {
    'N-Pose',         'trial0002', 'standing'
    'straightrunning','trial0025', 'straightrunning'
    'curvedrunning',  'trial0100', 'curvedrunning'
    'vcut',           'trial0122', 'vcut'
};

count = 0;
for i = 1:size(movements, 1)
    movDir = movements{i, 1};
    trial  = movements{i, 2};
    name   = movements{i, 3};
    outFile = fullfile(dstDir, sprintf('%s_%s.mat', name, trial));
    if exist(outFile, 'file')
        fprintf('Output file already exists, skipping: %s\n', outFile);
        continue;
    end

    try
        measuredFile = fullfile(srcDir, movDir, [trial '_DataStructMeasured.mat']);
        inverseFile  = fullfile(srcDir, movDir, [trial '_DataStructInverse.mat']);
        S = struct();
        S.movement = name;
        S.trial = trial;
        S.measured = convert_datastruct(load(measuredFile).dataStruct);
        S.inverse  = convert_datastruct(load(inverseFile).dataStruct);
    catch e
        fprintf('Failed to convert %s/%s: %s\n', movDir, trial, e.message);
        continue;
    end

    save(outFile, '-struct', 'S', '-v7');
    fprintf('Saved converted file: %s\n', outFile);
    count = count + 1;
end

fprintf('Conversion complete -- %d files written to %s\n', count, dstDir);
end


function out = convert_datastruct(ds)
    % Scalar fields (char/numeric) survive save() as-is; studyDate is a
    % datetime object (not -v7 saveable) and movementEvents/variables are
    % MATLAB tables (MCOS) -- all three need explicit flattening.
    out = struct();
    out.studyName = ds.studyName;
    out.studyDate = char(ds.studyDate);
    out.participantName = ds.participantName;
    out.participantSex = ds.participantSex;
    out.participantAge = ds.participantAge;
    out.participantMass = ds.participantMass;
    out.trialList = ds.trialList;
    out.movementType = ds.movementType;
    out.movementDescription = ds.movementDescription;
    out.isSymmetric = ds.isSymmetric;
    out.movementEvents = table2struct(ds.movementEvents, 'ToScalar', true);
    out.variables = table2struct(ds.variables, 'ToScalar', true);
end
