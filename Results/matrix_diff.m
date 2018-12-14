close all, clear all
NK = load('maha_I_init_train_param_');
RBF = load('RBF_maha_I_init_train_param_');
Poly = load('poly_maha_I_init_train_param_');

% diff = squeeze(NK.L(1, :, :))*squeeze(NK.L(1, :, :))' - ...
% squeeze(RBF.L(1, :, :))*squeeze(RBF.L(1, :, :))';
diff = squeeze(NK.L(1, :, :))*squeeze(NK.L(1, :, :))' - ...
eye(size(NK.L(1, :, :), 2));

%diff(logical(eye(size(diff, 1)))) = 0;
imagesc(abs(diff))
colorbar