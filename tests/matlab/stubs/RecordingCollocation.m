classdef RecordingCollocation < handle
    properties
        variables = struct()
        objectives = {}
    end
    methods
        function addOptimVar(obj, name, lower, upper, varargin)
            assert(isequal(size(lower), size(upper)), 'Bounds must have equal shape.');
            obj.variables.(name) = [{lower, upper}, varargin];
        end
        function addObjective(obj, varargin)
            obj.objectives{end + 1} = varargin;
        end
        function makeinitialguess(~, ~)
        end
        function addConstraint(~, varargin)
        end
    end
end
