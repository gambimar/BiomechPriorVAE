function convert_markertracking_to_plain(srcDir, dstDir)
% convert_markertracking_to_plain Convert MarkerTracking3D Result .mat files to plain MAT files
%
% Usage (from terminal, run on the workstation -- see AGENTS.md, the local
% mac's copy of the toolbox has an unrelated MEX-cache bug that thrashes a
% multi-hour rebuild on every Gait3d construction):
% LM_PROJECT=iwse matlab -batch "addpath(genpath('BioMAC-Sim-Toolbox/src')); convert_markertracking_to_plain('BioMAC-Sim-Toolbox/results/MarkerTracking3D/Participant_02','results_sim/MarkerTracking3D/Participant_02_converted')"
%
% Each raw file holds a `result` Result object (MCOS) from a Nitschke et al.
% 2023 marker-tracking OCP. load() reconstructs the Gait3d model inside it,
% so this needs the toolbox on the path (see ExampleScripts/+MarkerTracking3D/
% scriptMarkerTracking.m for the pattern this mirrors). Per file we pull the
% actually-tracked marker set from the trackMarker objective term (it differs
% per sparsity condition -- that is the independent variable of interest) and
% call Collocation.extractData() to get every simulated + tracked channel as
% one table, which is flattened into a plain struct and saved -v7 so scipy
% can read it directly (mirrors scripts/convert_mat_to_plain.m's convention:
% source dir -> separate converted dir, never in-place, skip existing output).

if nargin < 1 || isempty(srcDir)
    srcDir = 'BioMAC-Sim-Toolbox/results/MarkerTracking3D/Participant_02';
end
if nargin < 2 || isempty(dstDir)
    dstDir = 'results_sim/MarkerTracking3D/Participant_02_converted';
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

        objTermNames = {result.problem.objectiveTerms.name};
        idxMarker = find(strcmp(objTermNames, 'trackMarker'), 1);
        markerTable = result.problem.objectiveTerms(idxMarker).varargin{1}.variables;

        settings = struct();
        settings.marker = markerTable;
        simVarTable = result.problem.extractData(result.X, settings);

        S = struct();
        S.converged = result.converged;
        S.info = result.info;
        S.gitHashString = result.gitHashString;
        % Rows come in triplets (one per x/y/z direction) per tracked marker --
        % this collapses that back to the sparsity condition's marker list.
        S.tracked_markers = unique(markerTable.name, 'stable');
        S.simVarTable = table2struct(simVarTable, 'ToScalar', true);
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
