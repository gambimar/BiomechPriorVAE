classdef RecordingMomentModel < handle
    properties
        calls = 0
        coefficients = ones(1, 33)
    end
    methods
        function [moments, dMdx, dMdu] = getJointmoments(obj, states, controls)
            obj.calls = obj.calls + 1;
            moments = obj.coefficients' * (states(1) + controls(1));
            dMdx = zeros(numel(states), 33);
            dMdx(1,:) = obj.coefficients;
            dMdu = zeros(numel(controls), 33);
            dMdu(1,:) = obj.coefficients;
        end
    end
end
