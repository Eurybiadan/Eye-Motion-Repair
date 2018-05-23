function [ static_grid_distortion ] = Static_Distortion_Repair( horizontal_fringes_indices_minima )
% [ static_grid_distortion ] = Static_Distortion_Repair( horizontal_fringes_indices_minima )
%   Robert F Cooper, 2017-07-13
%
% This function estimates the residual static distortion not currently
% captured by the dewarping software.


fitstartind=10; % The point at which we'll start fitting (as we know the beginning is distorted)

horizontal_fringes_indices_minima=horizontal_fringes_indices_minima';

xvals = (1:length(horizontal_fringes_indices_minima))';
% Determine the linreg fit coefficients (X\y)
coeff = [ones(size(xvals(fitstartind:end))) xvals(fitstartind:end)] \ horizontal_fringes_indices_minima(fitstartind:end);

fringe_reg = coeff(2).*xvals + coeff(1);

% figure; plot(xvals,horizontal_fringes_indices_minima,'r',xvals,fringe_reg,'k'); 


residuals = (fringe_reg-horizontal_fringes_indices_minima);

p=fit(horizontal_fringes_indices_minima,residuals,'exp2');

% Set for how far into the fringes you want to attempt to repair distortion
dist_fix_rng = 1000;

static_grid_distortion = zeros(dist_fix_rng,1);
for i=1:dist_fix_rng
	static_grid_distortion(i) = -p(i);
end

% figure; plot(1:dist_fix_rng, -static_grid_distortion,horizontal_fringes_indices_minima,residuals);
end

