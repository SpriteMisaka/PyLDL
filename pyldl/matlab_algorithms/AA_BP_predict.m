net=newff(minmax(features'),[24,60,size(labels,2)],{'tansig','tansig','purelin'},'traingd');

net.IW = reshape(IW, 3, 1);
net.LW = transpose(reshape(LW, 3, 3));
net.b = reshape(b, 3, 1);

preDistribution = aabpPredict(net, testFeatures);
