%% WGAN-GP code
% Author: Dr. Weilu Tian & Prof. Peixue Ling
% Shandong University, Qingdao, China
% April 2024
% Using the Octane NIR dataset provided by MathWorks as a demo

%% Clear the workspace
clc;
clear;
close all;

%% Loading data
load spectra
spectroscopicData = single(NIR);
octaneValues = single(octane);

%% Data preprocessing
minSpec = min(spectroscopicData(:));
maxSpec = max(spectroscopicData(:));
normalizedSpectra = 2 * (spectroscopicData - minSpec) / (maxSpec - minSpec) - 1;
minOct = min(octaneValues(:));
maxOct = max(octaneValues(:));
normalizedOctane = (octaneValues - minOct) / (maxOct - minOct);
[numSamples, spectrumLength] = size(normalizedSpectra);

%% Defining the generator network
latentDim = 100;
octaneDim = 1;
layersG = [
    featureInputLayer(latentDim + octaneDim, 'Name', 'generator_input')
    fullyConnectedLayer(256, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(512, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(spectrumLength, 'Name', 'fc3')
    tanhLayer('Name', 'generator_output')
];

dlnetG = dlnetwork(layersG);

%% Defining the discriminator network
inputSpectra = featureInputLayer(spectrumLength,'Name','spectrum_input');
inputOctane = featureInputLayer(octaneDim,'Name','octane_input');
concatLayer = concatenationLayer(1,2,'Name','concat_inputs');

lgraphD = layerGraph();
lgraphD = addLayers(lgraphD, inputSpectra);
lgraphD = addLayers(lgraphD, inputOctane);
lgraphD = addLayers(lgraphD, concatLayer);

discriminatorLayers = [
    fullyConnectedLayer(512, 'Name', 'disc_fc1')
    leakyReluLayer(0.2, 'Name', 'disc_lrelu1')
    fullyConnectedLayer(256, 'Name', 'disc_fc2')
    leakyReluLayer(0.2, 'Name', 'disc_lrelu2')
    fullyConnectedLayer(1, 'Name', 'discriminator_output')
];

lgraphD = addLayers(lgraphD, discriminatorLayers);
lgraphD = connectLayers(lgraphD,'spectrum_input','concat_inputs/in1');
lgraphD = connectLayers(lgraphD,'octane_input','concat_inputs/in2');
lgraphD = connectLayers(lgraphD,'concat_inputs','disc_fc1');
dlnetD = dlnetwork(lgraphD);

%% Training parameter settings
numEpochs = 1000;
miniBatchSize = 10;
learnRateG = 1e-4;
learnRateD = 1e-4;
nCritic = 5;
lambdaGP = 10;
beta1 = 0.5; 
beta2 = 0.999;

%% Initialize optimizer state
avgGradG = [];
avgGradSqG = [];
avgGradD = [];
avgGradSqD = [];

for epoch = 1:numEpochs
    idx = randperm(numSamples);
    currentSpecs = normalizedSpectra(idx,:);
    currentOcts = normalizedOctane(idx,:);
    totalBatch = floor(numSamples / miniBatchSize);
    
    for i = 1:totalBatch
        batchIdx = (i-1)*miniBatchSize+1 : i*miniBatchSize;
        X = currentSpecs(batchIdx,:);
        Y = currentOcts(batchIdx,:);
        dlX = dlarray(X','CB');
        dlY = dlarray(Y','CB');
        
        % Training the Discriminator
        for j = 1:nCritic
            [gradientsD, lossD] = dlfeval(@modelLossD_WGANGP, dlnetG, dlnetD, dlX, dlY, lambdaGP, latentDim);
            [dlnetD, avgGradD, avgGradSqD] = adamupdate(dlnetD, gradientsD, avgGradD, avgGradSqD, epoch*totalBatch*nCritic + (i-1)*nCritic + j, learnRateD, beta1, beta2);
        end
        
        % Training the Generator
        [gradientsG, lossG] = dlfeval(@modelLossG_WGANGP, dlnetG, dlnetD, dlY, latentDim);
        [dlnetG, avgGradG, avgGradSqG] = adamupdate(dlnetG, gradientsG, avgGradG, avgGradSqG, epoch*totalBatch + i, learnRateG, beta1, beta2);

         if mod(i, 10) == 0
         end
    end
end

%% Demo of generating new spectra
numNew = 5;
targetOrig = [82, 85, 87, 89, 90];
targetNorm = (targetOrig - minOct) / (maxOct - minOct);
newZ = randn(latentDim, numNew, 'single');
dlNewZ = dlarray(newZ, 'CB');
dlNewOct = dlarray(targetNorm, 'CB');
dlInputNew = cat(1, dlNewZ, dlNewOct);
genNewSpec = extractdata(forward(dlnetG, dlInputNew));

% Convert the resulting spectra back to the original range
genNewSpecOrig = (genNewSpec + 1)/2*(maxSpec-minSpec)+minSpec;

%% Loss function definition
function [gradientsD, lossD, D_real, D_fake] = modelLossD_WGANGP(dlnetG, dlnetD, dlRealSpectra, dlRealOctane, lambdaGP, latentDim)
    D_real = forward(dlnetD, dlRealSpectra, dlRealOctane);
    batchSize = size(dlRealSpectra, 2);
    Z = randn(latentDim, batchSize, 'single');
    dlZ = dlarray(Z, 'CB');
    randIdx = randperm(batchSize);
    dlGenOct = dlarray(extractdata(dlRealOctane(:, randIdx)), 'CB');
    dlGenInput = cat(1, dlZ, dlGenOct);
    dlXGen = forward(dlnetG, dlGenInput);
    D_fake = forward(dlnetD, dlXGen, dlGenOct);
    % Wasserstein loss
    lossD_un = mean(D_fake) - mean(D_real);
    % Gradient penalty
    epsilon = rand(1, batchSize, 'single');
    eps = dlarray(epsilon, 'CB');
    dlXInt = eps .* dlRealSpectra + (1-eps) .* dlXGen;
    dlOInt = eps .* dlRealOctane + (1-eps) .* dlGenOct;
    interpOutput = forward(dlnetD, dlXInt, dlOInt);
    interpSum = sum(interpOutput, 'all');
    [gradSpec, gradOct] = dlgradient(interpSum, dlXInt, dlOInt, 'EnableHigherDerivatives', true);
    gradNorms = sqrt(sum(gradSpec.^2, 1) + sum(gradOct.^2, 1));
    gradPen = mean((gradNorms - 1).^2);
    % Total loss
    lossD = lossD_un + lambdaGP * gradPen;
    gradientsD = dlgradient(lossD, dlnetD.Learnables);
end

function [gradientsG, lossG] = modelLossG_WGANGP(dlnetG, dlnetD, dlRealOctane, latentDim)
    batchSize = size(dlRealOctane, 2);
    Z = randn(latentDim, batchSize, 'single');
    dlZ = dlarray(Z, 'CB');
    randIdx = randperm(batchSize);
    dlGenOct = dlarray(extractdata(dlRealOctane(:, randIdx)), 'CB');
    dlInput = cat(1, dlZ, dlGenOct);
    dlXGen = forward(dlnetG, dlInput);
    D_fake = forward(dlnetD, dlXGen, dlGenOct);
    lossG = -mean(D_fake);
    gradientsG = dlgradient(lossG, dlnetG.Learnables);
end
