% rotational invariant 2d welch win
function phiJ=compute_phiJ_hanning(M)
    addpath ./window2
    phiJ=window2(M,M,@hann);
end
