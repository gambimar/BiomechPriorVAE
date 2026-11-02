function tests = test_markertracking_paths
tests = functiontests(localfunctions);
end

function testTrailingSeparatorAndAbsoluteDestination(testCase)
testDir = fileparts(mfilename('fullpath'));
oldPath = path;
pathCleanup = onCleanup(@() path(oldPath));
addpath(fullfile(testDir, '..', '..', 'scripts'));
workDir = tempname;
mkdir(workDir);
directoryCleanup = onCleanup(@() rmdir(workDir, 's'));
srcDir = fullfile(workDir, 'source');
dstDir = fullfile(workDir, 'destination');
mkdir(fullfile(srcDir, 'nested'));

markerTable = table({'marker'}, 'VariableNames', {'name'});
term = struct('name', 'trackMarker', 'varargin', {{struct('variables', markerTable)}});
result = struct('converged', true, 'info', struct(), 'gitHashString', '', 'X', 0);
result.problem = struct('objectiveTerms', term, ...
    'extractData', @(varargin) table({'channel'}, 'VariableNames', {'name'}));
save(fullfile(srcDir, 'nested', 'raw.mat'), 'result');

convert_markertracking_to_plain([srcDir filesep], dstDir);
outFile = fullfile(dstDir, 'nested', 'raw.mat');
verifyTrue(testCase, isfile(outFile));
converted = load(outFile);
verifyTrue(testCase, converted.converged);
verifyEqual(testCase, converted.tracked_markers, {'marker'});
verifyTrue(testCase, isfield(load(fullfile(srcDir, 'nested', 'raw.mat')), 'result'));
end
