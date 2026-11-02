function output = vaeStateGradient(X, gradient, currentIdx, idx, momentScale, dMdx, dMdu, useMoments)
% Map feature gradients back to the collocation decision vector.
nCurrent = size(currentIdx, 1);
nNodes = size(currentIdx, 2);
if ~isequal(size(gradient), [nNodes nCurrent + 33])
    error('vaeReconstructionTerm:InvalidGradient', 'Python returned an unexpected gradient shape.');
end
output = zeros(size(X));
output(currentIdx) = gradient(:,1:nCurrent)';
if useMoments
    dVAEdM = gradient(:,nCurrent + 1:end);
    for i = 1:nNodes
        output(idx.states(:,i)) = output(idx.states(:,i)) + dMdx{i} * dVAEdM(i,:)'*momentScale;
        output(idx.controls(:,i)) = output(idx.controls(:,i)) + dMdu{i} * dVAEdM(i,:)'*momentScale;
    end
end
end
