function [target, gradient] = BFGS_Process(weights)

    global features
    global labels
    
    modProb = exp(features * weights);
    sumProb = sum(modProb, 2);
    modProb = modProb ./ (repmat(sumProb, [1 size(modProb, 2)]));
    
    target = -sum(sum(labels.*log(modProb)));
    gradient = features'*(modProb - labels);

end
