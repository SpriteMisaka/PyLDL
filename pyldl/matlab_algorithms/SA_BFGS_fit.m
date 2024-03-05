item = eye(size(features, 2), size(labels, 2));
[weights, fval] = bfgslldTrain(@BFGS_Process,item);
