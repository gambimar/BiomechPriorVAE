function [indices, momentOrder] = vaeStateIndices(model)
% Arrange BioMAC state rows in the feature order expected by the Python prior.
jointNames = {'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', ...
    'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r', ...
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', ...
    'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l', ...
    'lumbar_extension', 'lumbar_bending', 'lumbar_rotation', ...
    'arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_r', 'pro_sup_r', ...
    'arm_flex_l', 'arm_add_l', 'arm_rot_l', 'elbow_flex_l', 'pro_sup_l'};
qAll = model.extractState('q');
if numel(qAll) ~= 33 || numel(model.extractState('qdot')) ~= 33
    error('vaeStateIndices:UnsupportedJoints', 'The prior requires the 33-DOF BioMAC gait model.');
end
q = model.extractState('q', jointNames);
qdot = model.extractState('qdot', jointNames);
[found, jointOrder] = ismember(q(:), qAll(:));
if ~all(found)
    error('vaeStateIndices:InvalidStates', 'Named joints are missing from the model state metadata.');
end
momentOrder = [setdiff((1:33)', jointOrder, 'stable'); jointOrder];
parts = {q(:), qdot(:)};
contactNames = model.CPs.Properties.RowNames;
rightNames = contactNames(endsWith(contactNames, '_r'));
leftNames = contactNames(endsWith(contactNames, '_l'));
if ~ismember(numel(rightNames), [6 8]) || numel(leftNames) ~= numel(rightNames) || ...
        numel(rightNames) + numel(leftNames) ~= numel(contactNames)
    error('vaeStateIndices:UnsupportedContacts', 'The prior requires six or eight contacts on each foot.');
end
axes = {'Fx', 'Fy', 'Fz'};
sides = {rightNames, leftNames};
for iSide = 1:2
    for iAxis = 1:3
        rows = model.extractState(axes{iAxis}, sides{iSide});
        parts{end + 1} = rows(:);
    end
end
indices = vertcat(parts{:});
if numel(q) ~= 27 || numel(qdot) ~= 27 || numel(unique(indices)) ~= numel(indices) || ...
        any(indices < 1) || any(indices > model.nStates)
    error('vaeStateIndices:InvalidStates', 'Model state metadata does not match the prior feature layout.');
end
end
