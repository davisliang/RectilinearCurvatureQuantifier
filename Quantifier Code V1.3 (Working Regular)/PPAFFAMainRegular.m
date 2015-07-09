%6/18/2015, Davis Liang, Version 1.2 
%Variable control center for PPA Model 
close all;
%% variables
preprocess = false;
oneLayer = false;
runFinalTest = true;
activation = 'sigmoid'; %'sigmoid';
train = true;

numberIterations = 100;
numhid = 20;
targetLength = 1; 
numCategories = 2;
numTrain = 90;
numTest = 8; 
imPerFolder = numTrain + numTest;
S = 0; M = 0;

preprocessLoc = '/Users/davisliang/Desktop/DataSets/FaceSceneSet/';    %image location
matFileLoc = '/Users/davisliang/Desktop/'; %matFileLocation
matFile = 'PrepFaceVsScene_Tr90_Te8_RegularGabor.mat'; %matfile name
fileType = '.pgm';
testFile = '1288.pgm';

%% preprocessor
if(preprocess)
    PreprocessData(preprocessLoc, numTrain, numTest, matFile, fileType, numCategories);
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
    outputs = PPAFinalTestRegular(activation, testFile);
end



