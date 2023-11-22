model.Sigma = reshape(model.Sigma, size(model.Sigma, 2), 1);
preDistribution = ptbayesPredict(model, features);
