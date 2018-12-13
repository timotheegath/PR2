clear all, close all
load('kernel_Maha_RBFslow_eye-init2')
plot(test_mAp)
figure
plot(train_mAp)
figure
plot(loss)
load('kernel_Maha_RBFslow_eye-init_param_')

figure
imagesc(squeeze(L_matrix(3, :, :)))
colorbar