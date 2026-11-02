function tests = test_vae_features
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
testDir = fileparts(mfilename('fullpath'));
testCase.TestData.originalPath = path;
addpath(fullfile(testDir, '..', '..'));
addpath(fullfile(testDir, 'stubs'));
end

function teardownOnce(testCase)
path(testCase.TestData.originalPath);
end

function testSixContactsAndArbitraryMuscleCount(testCase)
checkLayout(testCase, 6, 5);
end

function testEightContactsAndArbitraryMuscleCount(testCase)
checkLayout(testCase, 8, 17);
end

function testScrambledModelJointOrder(testCase)
model = modelFixture(6, 5, true);
[indices, momentOrder] = vaeStateIndices(model);
verifyEqual(testCase, indices(1:27), (27:-1:1)');
verifyEqual(testCase, momentOrder, [28:33, 27:-1:1]');
momentModel = RecordingMomentModel();
momentModel.coefficients = 1:33;
[moments, dMdx, dMdu] = vaeJointMoments(momentModel, ones(4, 1), 2*ones(3, 1), true, true, momentOrder);
verifyEqual(testCase, moments, 3*momentOrder);
verifyEqual(testCase, dMdx{1}(1,:), momentOrder');
verifyEqual(testCase, dMdu{1}(1,:), momentOrder');
end

function testExplicitFeatureLayoutsResolveAmbiguousWidths(testCase)
[useMoments, stateKeys] = vaeFeatureLayout(struct('numDofs', 46));
verifyFalse(testCase, useMoments);
verifyEmpty(testCase, stateKeys);
[useMoments, stateKeys] = vaeFeatureLayout(struct('numDofs', 46, 'stateKeys', {{'pos', 'tau'}}));
verifyTrue(testCase, useMoments);
verifyEqual(testCase, stateKeys, {'pos', 'tau'});
[useMoments, ~] = vaeFeatureLayout(struct('numDofs', 27, 'stateKeys', {{'force', 'pos'}}));
verifyFalse(testCase, useMoments);
end

function checkLayout(testCase, nFootContacts, nMuscles)
[model, base] = modelFixture(nFootContacts, nMuscles);
indices = vaeStateIndices(model);
rightFx = base + (7:12:12*nFootContacts - 5);
leftFx = base + (1:12:12*nFootContacts - 11);
expected = [7:33, 40:66, rightFx, rightFx + 1, rightFx + 2, leftFx, leftFx + 1, leftFx + 2]';
verifyEqual(testCase, indices, expected);
verifyEqual(testCase, numel(indices) + 33, 87 + 6*nFootContacts);
end

function testUnsupportedContactCount(testCase)
model = modelFixture(7, 5);
verifyError(testCase, @() vaeStateIndices(model), 'vaeStateIndices:UnsupportedContacts');
end

function testMomentFreePriorDoesNotEvaluateMoments(testCase)
model = RecordingMomentModel();
[moments, dMdx, dMdu] = vaeJointMoments(model, ones(4, 2), ones(3, 2), false, true);
verifyEqual(testCase, model.calls, 0);
verifyEqual(testCase, moments, zeros(33, 2));
verifyEmpty(testCase, dMdx{1});
verifyEmpty(testCase, dMdu{1});
end

function testMomentPriorPassesControlsAndDerivatives(testCase)
model = RecordingMomentModel();
[moments, dMdx, dMdu] = vaeJointMoments(model, ones(4, 2), 2*ones(3, 2), true, true);
verifyEqual(testCase, model.calls, 2);
verifyEqual(testCase, moments, 3*ones(33, 2));
verifyEqual(testCase, dMdx{1}(1,:), ones(1, 33));
verifyEqual(testCase, dMdu{1}(1,:), ones(1, 33));
end

function testSixContactGradientMapping(testCase)
checkGradient(testCase, 6);
end

function testEightContactGradientMapping(testCase)
checkGradient(testCase, 8);
end

function checkGradient(testCase, nFootContacts)
model = modelFixture(nFootContacts, 5);
rows = vaeStateIndices(model);
nStates = model.nStates;
idx.states = reshape(1:2*nStates, nStates, 2);
idx.controls = reshape(2*nStates + (1:6), 3, 2);
currentIdx = idx.states(rows,:);
X = zeros(2*nStates + 6, 1);
gradient = reshape(1:2*(numel(rows) + 33), 2, []);
output = vaeStateGradient(X, gradient, currentIdx, idx, 0, {}, {}, false);
verifyEqual(testCase, output(currentIdx), gradient(:,1:numel(rows))');
verifyEqual(testCase, output(idx.controls), zeros(3, 2));

modelMoments = RecordingMomentModel();
[~, dMdx, dMdu] = vaeJointMoments(modelMoments, ones(nStates, 2), ones(3, 2), true, true);
outputWithMoments = vaeStateGradient(X, gradient, currentIdx, idx, 0.25, dMdx, dMdu, true);
expectedMomentChain = 0.25*sum(gradient(:,numel(rows) + 1:end), 2)';
verifyEqual(testCase, outputWithMoments(idx.controls(1,:)), expectedMomentChain');
verifyEqual(testCase, outputWithMoments(idx.states(1,:)), expectedMomentChain');
verifyError(testCase, @() vaeStateGradient(X, gradient(:,1:end - 1), currentIdx, idx, 0, {}, {}, false), ...
    'vaeReconstructionTerm:InvalidGradient');
end

function [model, base] = modelFixture(nFootContacts, nMuscles, scrambleJoints)
jointNames = {'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', ...
    'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r', ...
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', ...
    'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l', ...
    'lumbar_extension', 'lumbar_bending', 'lumbar_rotation', ...
    'arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_r', 'pro_sup_r', ...
    'arm_flex_l', 'arm_add_l', 'arm_rot_l', 'elbow_flex_l', 'pro_sup_l'};
base = 66 + 2*nMuscles;
nStates = base + 12*nFootContacts;
types = repmat({'other'}, nStates, 1);
names = repmat({''}, nStates, 1);
types(1:33) = {'q'};
types(34:66) = {'qdot'};
names(7:33) = jointNames;
names(40:66) = jointNames;
if nargin >= 3 && scrambleJoints
    names(1:33) = names(33:-1:1);
    names(34:66) = names(66:-1:34);
end
contactNames = cell(2*nFootContacts, 1);
for i = 1:nFootContacts
    contactNames{2*i - 1} = sprintf('point%d_l', i);
    contactNames{2*i} = sprintf('point%d_r', i);
end
axes = {'Fx', 'Fy', 'Fz', 'xc', 'yc', 'zc'};
for i = 1:numel(contactNames)
    rows = base + 6*(i - 1) + (1:6);
    types(rows) = axes;
    names(rows) = contactNames(i);
end
model = struct();
model.nStates = nStates;
model.CPs = table(zeros(numel(contactNames), 1), 'RowNames', contactNames);
model.extractState = @(type, varargin) lookupState(types, names, type, varargin{:});
end

function rows = lookupState(types, names, type, requestedNames)
if nargin < 4
    rows = find(strcmp(types, type));
else
    rows = zeros(size(requestedNames));
    for i = 1:numel(requestedNames)
        rows(i) = find(strcmp(types, type) & strcmp(names, requestedNames{i}));
    end
end
end
