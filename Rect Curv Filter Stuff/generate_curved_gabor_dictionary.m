function [Dictionary] = generate_curved_gabor_dictionary(numCurves, numOrientations, numScales, frequency, maxSize)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%generally we want a frequency of 0.5
sigmaone=0.3;
sigmatwo=4;

for i = 1:numScales
    sizeMod(i) = maxSize;
    maxSize = (maxSize+3)/2
end
%for scales = 5 and maxSize = 99, then sizeMod(1)=99, sizeMod(2)=51...
%sizeMod(5)=9

setMinCurvature = 0.0125;
for i = 1:numCurves
    curve(i) = setMinCurvature;
    setMinCurvature = setMinCurvature*1.4;
end
%generally we want numCurves = 8

setMinOrientations = 2*pi/numOrientations;
for i = 1:numOrientations
    orientations(i) = setMinOrientations;
    setMinOrientations = setMinOrientations + 2*pi/numOrientations;
end
%generally we want 8 orientations and multiples of pi/4 as orientations    
    
for s = 1:size(sizeMod,2) %for each size
    sigmaone = sigmaone/1.6;
    sigmatwo = sigmatwo/1.6;
    
    for c = 1:size(curve,2) %for each curvature
        for o = 1:size(orientations,2) %for each orientation 
            len=-(sizeMod(s)-1)/2:1:(sizeMod(s)-1)/2;  
            temp = banana_filter(frequency,orientations(o), curve(c), len,len, sigmaone, sigmatwo);       
            Dictionary{s}{o,c} = temp;
        end
    end
end


end

