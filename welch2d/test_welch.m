imgs = randn(10,64,64); % 10 white noise images
wJ = 4; % window size is 2^wJ, should be smaller than image size
pos = compute_power_spectrum_welch(imgs,wJ);
[Spos,Vpos,Kpos] = mySpectre2D(pos);
plot(Kpos,Spos)
xlabel('freq mode')
ylabel('radial power spectrum')

% imagesc(fftshift(pos),[0,2])
% colorbar 
% colormap gray