clear all, close all
load('poly_simi_init_train')
% test_mAp = [back.test_mAp; next.test_mAp] * 100
% train_mAp = [back.train_mAp; next.train_mAp]
% loss = [back.loss; next.loss]
figure
plot(test_mAp*100)

grid on
grid minor
xlabel('Epoch')
ylabel('mAp (%)')
legend('No kernel', 'RBF kernel', 'Polynomial kernel')