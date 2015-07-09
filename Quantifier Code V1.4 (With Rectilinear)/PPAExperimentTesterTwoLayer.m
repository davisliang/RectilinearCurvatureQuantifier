%4/22/2015, Davis Liang
function [finalOut] = PPAExperimentTesterTwoLayer(whi, woh, testingData, activation, targetCategory)
numberWrong = 0;
Error = 0;
bias = [1];
%name identifying constants used in learning algorithm 

finalOut = [];

%testing code here. Essentially, take the testingData, multiply with the
%weight matrix and then find the error. this is 1 iteration!

testInput = [bias;testingData];
               

neti = [(whi(:,:)*testInput)];  %net input to either hidden unit     
hout = [1./(1+exp(-neti))];
            
%output layer
h_layer = [bias; hout];         %complete hidden layer
neto = woh*[h_layer];           %net input to each output unit                  
            
if(strcmp(activation,'sigmoid'))
    out = (1./(1+exp(-neto)));
else
    out = exp(neto)./sum(exp(neto));
end


finalOut = out;