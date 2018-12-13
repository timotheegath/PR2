close all, clear all
load('baseline')
keys = {'Bilinear Similarity', 'Cross-correlation', 'Cosine Similarity' ...
    'Mahalanobis distance', 'Minkovsky metric, p=1' ...
    'Minkovsky metric, p=2'};
results = {bi, cc, cos, mah, mink1, mink2};
rank = 1:14;
figure
hold on
grid on
grid minor
for i = 1:length(results)
    
    plot(rank, results{i}.score * 100)
    title('Baseline mAps varying with rank')
    legend(keys)
    ylabel('%')
    xlabel('Rank')
    
    
end
