function tests = test_running3D
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
testDir = fileparts(mfilename('fullpath'));
testCase.TestData.originalPath = path;
addpath(fullfile(testDir, '..', '..', 'scripts', 'func'));
addpath(fullfile(testDir, '..', '..'));
addpath(fullfile(testDir, 'stubs'));
end

function teardownOnce(testCase)
path(testCase.TestData.originalPath);
end

function testOmittedPrior(testCase)
[model, W] = fixture();
problem = running3D(model, 0, 'standing.mat', 'running.mat', 3, true, W, 1.33);
verifyEmpty(testCase, problem.objectives);
end

function testEmptyPrior(testCase)
[model, W] = fixture();
problem = running3D(model, 0, 'standing.mat', 'running.mat', 3, true, W, 1.33, []);
verifyEmpty(testCase, problem.objectives);
end

function testZeroWeightPrior(testCase)
[model, W] = fixture();
vae = struct('modelPath', 'model.pth', 'pythonPath', '.', 'numDofs', 50, 'latentDim', 24, 'weight', 0);
problem = running3D(model, 0, 'standing.mat', 'running.mat', 3, true, W, 1.33, vae);
verifyEmpty(testCase, problem.objectives);
end

function testFreeSpeedBounds(testCase)
[model, W] = fixture();
W.speed = 1;
problem = running3D(model, 0, 'standing.mat', 'running.mat', 3, true, W, 0, []);
verifyEqual(testCase, problem.variables.speed{1}, [0.2; -0.01]);
verifyEqual(testCase, problem.variables.speed{2}, [1.8; 0.01]);
verifyEqual(testCase, problem.variables.speed{3}, [1; 0.003]);
end

function testUnsupportedDurationIsExplicit(testCase)
[model, W] = fixture();
W.dur = Inf;
verifyError(testCase, @() running3D(model, 0, '', '', 3, true, W, 1.33), 'running3D:UnsupportedDuration');
end

function testUnsupportedTrackingIsExplicit(testCase)
[model, W] = fixture();
W.track = 1;
verifyError(testCase, @() running3D(model, 0, '', '', 3, true, W, 1.33), 'running3D:UnsupportedTracking');
end

function testContactVariants(testCase)
stubDir = tempname;
mkdir(stubDir);
cleanup = onCleanup(@() cleanupStubs(stubDir));
variants = {'', '_smoothsphere', '_smoothsphere_softdamp2', '_smoothsphere_softdamp4', ...
    '_smoothsphere_stiffer2', '_smoothsphere_stiffer4', '_smoothsphere_nodamp', '_smoothsphere_stiff10x'};
for i = 1:numel(variants)
    name = ['Gait3d' variants{i}];
    fid = fopen(fullfile(stubDir, [name '.m']), 'w');
    fprintf(fid, 'function model = %s(modelFile)\nmodel = struct(''kind'', ''%s'', ''file'', modelFile);\nend\n', name, name);
    fclose(fid);
end
addpath(stubDir);
for i = 1:numel(variants)
    modelFile = ['model' variants{i} '.osim'];
    model = createGaitModel(modelFile);
    verifyEqual(testCase, model.kind, ['Gait3d' variants{i}]);
    verifyEqual(testCase, model.file, modelFile);
end
end

function cleanupStubs(stubDir)
rmpath(stubDir);
rmdir(stubDir, 's');
end

function [model, W] = fixture()
model = struct();
model.states = struct('xmin', -ones(4, 1), 'xmax', ones(4, 1));
model.controls = struct('xmin', zeros(2, 1), 'xmax', ones(2, 1));
model.constraints = struct('fmin', 0, 'fmax', 0);
model.osim = struct('file', 'synthetic.osim');
model.extractState = @(varargin) [];
W = struct('dur', 0, 'speed', Inf, 'reg', 0, 'effMuscles', 0, 'effTorques', 0, ...
    'effMusclesAct', 0, 'effMusclesTor', 0, 'metcost', 0, 'track', 0);
end
