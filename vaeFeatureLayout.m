function [useMoments, stateKeys] = vaeFeatureLayout(vaeParams)
% Explicit feature names resolve layouts with the same input width.
if isfield(vaeParams, 'stateKeys') && ~isempty(vaeParams.stateKeys)
    stateKeys = cellstr(string(vaeParams.stateKeys));
    useMoments = any(ismember(stateKeys, {'tau', 'M_ankle'}));
else
    stateKeys = {};
    useMoments = ~ismember(vaeParams.numDofs, [23 27 31 46 50 58]);
end
end
