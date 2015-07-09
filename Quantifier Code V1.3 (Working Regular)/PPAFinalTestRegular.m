function [outputs] = PPAFinalRegularTest(activation, filename)


%% variables
matFileLoc = '/Users/davisliang/Desktop/'; %matFileLocation
matFile = 'networkinfo.mat'; %matfile name
cd(matFileLoc);
matFile = load(matFile);
whi = matFile.weights.whi;
woh = matFile.weights.woh;

numTest = 1;
numTrain = 9;


preprocessLoc = '/Users/davisliang/Desktop/DataSets/PPARunDataSet/';
matFileLoc = '/Users/davisliang/Desktop/';
matFile = 'experimentalData.mat';

%% preprocessing
PreprocessDataExperiment(strcat(preprocessLoc, filename));
cd(matFileLoc);
matFile = load(matFile);
data = matFile.experimentData;
close all;
testingData = data.data;

%% experiment 

outputs = PPAExperimentTesterTwoLayer(whi, woh, testingData, activation)
