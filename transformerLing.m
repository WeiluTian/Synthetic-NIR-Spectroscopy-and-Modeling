%% Modified Transformer Network for Spectral Regression
% Author: Dr. Weilu Tian & Prof. Peixue Ling
% Shandong University, Qingdao, China
% April 2024
% Using the Octane NIR dataset provided by MathWorks as a demo

%% Clear workspace
clc;
clear;
close all;

%% Load spectral dataset
load spectra
Xdata = single(NIR);
Ydata = single(octane);
[numSamples, numFeatures] = size(Xdata);

% Prepare input as 1D sequences
X = cell(numSamples,1);
for i = 1:numSamples
    X{i} = reshape(Xdata(i,:), 1, []); % [features=1 x seqLen=401]
end

Y = Ydata;

%% Network Parameters
numHeads = 4;
embedDim = 128;
ffnDim = 256;
numEncoderLayers = 4;
dropoutProb = 0.05;

%% Input + Embedding Layers
layers = [
    sequenceInputLayer(1, 'Name','input')
    fullyConnectedLayer(embedDim, 'Name','embedding')
];

%% Positional Encoding Layer
posEnc = positionalEncodingLayer(numFeatures, embedDim, 'Name','posenc');
lgraph = layerGraph(layers);

%% Add the position encoding layer and connect
lgraph = addLayers(lgraph, posEnc);
lgraph = connectLayers(lgraph, 'embedding', 'posenc');
prevName = 'posenc';

%% Encoder Layers
for i = 1:numEncoderLayers
    mhaName = sprintf("mha%d", i);
    addNorm1 = sprintf("addnorm1_%d", i);
    ffnName = sprintf("ffn%d", i);
    addNorm2 = sprintf("addnorm2_%d", i);
    mha = attentionLayer(numHeads, 'Name', mhaName);
    lnorm1 = layerNormalizationLayer('Name', addNorm1);
    
    % Feed Forward Block
    ffn = [
        fullyConnectedLayer(ffnDim, 'Name', sprintf("fc1_%d", i))
        reluLayer('Name', sprintf("relu_%d", i))
        fullyConnectedLayer(embedDim, 'Name', sprintf("fc2_%d", i))
    ];
    
    lnorm2 = layerNormalizationLayer('Name', addNorm2);
    
    % Residual Additions
    add1 = additionLayer(2, 'Name', sprintf("add1_%d", i));
    add2 = additionLayer(2, 'Name', sprintf("add2_%d", i));
    
    % Add layers
    lgraph = addLayers(lgraph, mha);
    lgraph = addLayers(lgraph, lnorm1);
    lgraph = addLayers(lgraph, ffn);
    lgraph = addLayers(lgraph, lnorm2);
    lgraph = addLayers(lgraph, add1);
    lgraph = addLayers(lgraph, add2);
    
    % Connections
    lgraph = connectLayers(lgraph, prevName, mhaName + "/query");
    lgraph = connectLayers(lgraph, prevName, mhaName + "/key");
    lgraph = connectLayers(lgraph, prevName, mhaName + "/value");
    lgraph = connectLayers(lgraph, prevName, sprintf("add1_%d/in1", i));
    lgraph = connectLayers(lgraph, mhaName, sprintf("add1_%d/in2", i));
    lgraph = connectLayers(lgraph, sprintf("add1_%d", i), addNorm1);
    lgraph = connectLayers(lgraph, addNorm1, sprintf("fc1_%d", i));
    lgraph = connectLayers(lgraph, addNorm1, sprintf("add2_%d/in1", i));
    lgraph = connectLayers(lgraph, sprintf("fc2_%d", i), sprintf("add2_%d/in2", i));
    lgraph = connectLayers(lgraph, sprintf("add2_%d", i), addNorm2);
    
    prevName = addNorm2;
end

%% Output Head
postLayers = [
    globalAveragePooling1dLayer('Name','gap')
    fullyConnectedLayer(512, 'Name','fc_out1')
    reluLayer('Name','relu_out')
    fullyConnectedLayer(1, 'Name','fc_out2')
    regressionLayer('Name','regressionoutput')
];

lgraph = addLayers(lgraph, postLayers);
lgraph = connectLayers(lgraph, prevName, 'gap');
%% Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'MiniBatchSize', 6, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle','every-epoch', ...
    'Plots','none', ...
    'Verbose',false);

%% Train Network
net = trainNetwork(X, Y, lgraph, options);

%% Predict Y value
YPred = predict(net, X);


