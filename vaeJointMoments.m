function [moments, dMdx, dMdu] = vaeJointMoments(model, states, controls, useMoments, needGradient, momentOrder)
% Avoid joint-moment evaluations for priors that do not use those features.
nNodes = size(states, 2);
moments = zeros(33, nNodes);
dMdx = cell(1, nNodes);
dMdu = cell(1, nNodes);
if ~useMoments
    return;
end
if nargin < 6
    momentOrder = 1:33;
end
for i = 1:nNodes
    if needGradient
        [nodeMoments, nodeDx, nodeDu] = model.getJointmoments(states(:,i), controls(:,i));
        dMdx{i} = nodeDx(:, momentOrder);
        dMdu{i} = nodeDu(:, momentOrder);
    else
        nodeMoments = model.getJointmoments(states(:,i), controls(:,i));
    end
    moments(:,i) = nodeMoments(momentOrder);
end
end
