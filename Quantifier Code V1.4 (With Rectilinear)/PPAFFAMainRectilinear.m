%6/18/2015, Davis Liang, Version 1.2 
%Variable control center for PPA Model 
close all;
%% variables
preprocess = false;
oneLayer = false;
runFinalTest = false;
activation = 'sigmoid'; %'sigmoid';
train = true;

numberIterations = 100000;
numhid = 20;
targetLength = 1; 
numCategories = 2;
numTrain = 250;
numTest = 30; 
imPerFolder = numTrain + numTest;
S = 0; M = 0;

preprocessLoc = '/Users/davisliang/Desktop/DataSets/SharpSmoothSet/';    %image location
matFileLoc = '/Users/davisliang/Desktop/'; %matFileLocation
matFile = 'PrepSmoothSharp_Tr250_Te30_Regular_Gabor.mat'; %matfile name
fileType = '.jpg';
testFile = 'paintingface.jpg';

%% preprocessor
if(preprocess)
    preprocessDataPPA(preprocessLoc, numTrain, numTest, matFile, fileType, numCategories);
end

cd(matFileLoc);
matFile = load(matFile);
data = matFile.organizedData;
trainingData = data.train;
testingData = data.test;

%% neural network
if(train == true)
    display(['Training Network... ']);
    if(oneLayer == true)
        [learnedWeight, trainingError, trainingSteps, trainingAttempts]  = completeCompositeTrainer(trainingData, targetLength, testingData, activation, numberIterations) %output weight matrix with results too
        [testError, percentWrong, numberWrong] = completeCompositeTester(learnedWeight, testingData, activation)
    else
        [whi, woh, trainingError, trainingSteps, trainingAttempts]  = PPAFFATrainerTwoLayer(trainingData, targetLength, testingData, activation, numberIterations, numhid) %output weight matrix with results too
        [percentWrong, testError, numberWrong] = PPAFFATesterTwoLayer(whi, woh, testingData, activation)
    end
end

%% experiments
if(runFinalTest==true)
    outputs = PPAFinalTest(activation, testFile);
end



