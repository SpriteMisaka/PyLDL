para.minValue = 1e-7;
para.iter = 200;
para.minDiff = 1e-7;
para.regfactor = 0;

[weights] = iislldTrain(para, features, labels);
