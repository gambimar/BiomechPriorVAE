function jointVarTable = markertracking_joint_variable_table(model)
% markertracking_joint_variable_table Build a Gait3d.simuMarker-compatible
% variables table (type, name, segment, position, direction; 3 rows per
% joint, one per axis) for every anatomical joint center of `model`, for use
% as joint-center FK ground truth in the MarkerTracking3D MPJPE experiment
% (see scripts/convert_markertracking_joints_sim.m /
% scripts/convert_markertracking_reference_joints.m).
%
% Two special cases, verified interactively against Gait3d.showStick (the
% toolbox's own stick-figure renderer) on the workstation before writing
% this:
%   - ground_pelvis's parent_segment is 'ground', which has no entry in
%     Gait3d.getFkin's output (FK only covers the nSegments-1 non-ground
%     segments) -- it is the floating base, not a real anatomical joint, so
%     it is dropped from the joint set entirely.
%   - knee_r/knee_l have a flexion-dependent translating joint center in
%     this model: joints.location is a placeholder [0 0 0], and the real
%     location is a spline function of knee_angle via t1_coefs/t2_coefs.
%     Using the coefficients' constant term (column 5) as a static
%     approximation mirrors what Gait3d.showStick itself does for its own
%     femur polygon (see its `polygons{2}`/`polygons{7}` entries) -- same
%     approximation the toolbox already ships, not a new hack.

keep = ~strcmp(model.joints.parent_segment, 'ground');
jointsT = model.joints(keep, :);

isKnee = strcmp(jointsT.Properties.RowNames, 'knee_r') | strcmp(jointsT.Properties.RowNames, 'knee_l');
loc = jointsT.location;
loc(isKnee, :) = [jointsT.t1_coefs(isKnee, 5), jointsT.t2_coefs(isKnee, 5), zeros(sum(isKnee), 1)];

jointNames = jointsT.Properties.RowNames;
nJ = numel(jointNames);
name = repelem(jointNames, 3);
segment = repelem(jointsT.parent_segment, 3);
position = repelem(loc, 3, 1);
direction = repmat(eye(3), nJ, 1);
type = repmat({'joint'}, 3 * nJ, 1);
jointVarTable = table(type, name, segment, position, direction);
end
