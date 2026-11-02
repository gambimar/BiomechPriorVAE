function model = createGaitModel(modelFile)
% Use the same contact model for the standing guess and gait optimization.
if contains(modelFile, 'smoothsphere_softdamp2')
    model = Gait3d_smoothsphere_softdamp2(modelFile);
elseif contains(modelFile, 'smoothsphere_softdamp4')
    model = Gait3d_smoothsphere_softdamp4(modelFile);
elseif contains(modelFile, 'smoothsphere_stiffer2')
    model = Gait3d_smoothsphere_stiffer2(modelFile);
elseif contains(modelFile, 'smoothsphere_stiffer4')
    model = Gait3d_smoothsphere_stiffer4(modelFile);
elseif contains(modelFile, 'smoothsphere_nodamp')
    model = Gait3d_smoothsphere_nodamp(modelFile);
elseif contains(modelFile, 'smoothsphere_stiff10x')
    model = Gait3d_smoothsphere_stiff10x(modelFile);
elseif contains(modelFile, 'smoothsphere')
    model = Gait3d_smoothsphere(modelFile);
else
    model = Gait3d(modelFile);
end
end
