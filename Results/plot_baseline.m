close all, clear all
load('baseline')
keys = {'Bilinear Similarity', 'Cross-correlation', 'Cosine Similarity' ...
   'Minkowsky metric, p=1' ...
    'Minkowsky metric, p=2'};
results = {bi, cc, cos, mink1, mink2};
rank = 1:14;
figure
hold on
grid on
grid minor
for i = 1:length(results)
    
    plot(rank, results{i}.score * 100)
    %title('Baseline mAps varying with rank')
    legend(keys)
    ylabel('%')
    xlabel('Rank')
    
    
end
clear all
load('baseline_Maha')
keys = {'Covariance based Mahalanobis distance'...
    'Identity based Mahalanobis distance'};
results = {cov_Maha, eye_Maha};
rank=1:14;
figure
hold on
grid on
grid minor
for i = 1:length(results)
    plot(rank, results{i}.score * 100)
    %title('mAp varying with rank')
    legend(keys)
    ylabel('%')
    xlabel('Rank')
end
