function [whi, woh, trainingError, steps, attempts] = PPAFFATrainerTwoLayer(trainingData, targlen, testingData, activation,numIter, numhid)
%@author Davis Liang 
%@date 5/23/15
%@title Generalized Two Layer Network(matfile, .m)
%@note engine for holistic face processing. (see completeCompositeCode)
%@git davisliang

%% Preprocessing & Setup

num_category = size(trainingData,2);
num_trainImagesPerCategory = size(trainingData{1},2);
image_size = size(trainingData{1}(:,1),1);
totalImages = num_category * num_trainImagesPerCategory;

inlen = image_size;

%sd_in = std(input(:)); %standard deviation of the input matrix
%avg_in = sum(input)/insizeinfo; %average of the input matrix
%input = (1/sd_in)*(input-avg_in); %Z-SCORED
%target = target*2-1;    %set targets to be 1 or -1

whi = (normrnd(0,1/sqrt(inlen+1),[numhid,inlen+1]));    %randomized weight matrix for input layer to hidden layer                 
woh = (normrnd(0,1/sqrt(numhid+1),[targlen,numhid+1]));  %randomized weight matrix for hidden layer to output layer
bias = [1];     %bias
learnho = 0.001;
learnih = 0.001;

a = .9;         %momentum constant
dwoldh = 0;     %old weight change matrix for input to hidden units  
dwoldo = 0;     %old weight change matrix for hidden to output units

attempts = 1;   %shows how many attempts tried
steps = 0;      %how many steps it takes to learn?
run = 1;        %run, run, as fast as you can. please.

%% Stochastic Gradient Descent Code

Error = [];
TestError = [];
TestPercentWrong = [];
%mainprogram

while run
    steps = steps + 1;  %step increment
    trainingError = 0;              %total error
    
    for CategoryNum = (1:num_category)    %randomly permutate patterns  
        for imageNum = randperm(num_trainImagesPerCategory)
            %set up input
            input = trainingData{CategoryNum}(:,imageNum);
            in = [bias; input];
            
            %set up targets
            if(CategoryNum == 2)
                targ = 1; %is PPA/FFA active (Set 1 is faces, Set 2 is scenes and tools)
            else 
                targ = 0;
            end
            
            %hidden layer
            neti = [(whi(:,:)*in)];  %net input to either hidden unit     
            hout = [1./(1+exp(-neti))]; %hidden layer output
            
            %output layer
            h_layer = [bias; hout];         %complete hidden layer
            neto = woh*[h_layer];           %net input to each output unit
            if(strcmp(activation,'sigmoid'))
                out = 1./(1+exp(-neto));
                oprime = out.*(1-out);
            else
                out = exp(neto)./sum(exp(neto));
                oprime = 1;
            end
            
            
            %slopes
            hprime = hout.*(1-hout);    %slope of hidden units                   
           
        
            %deltas
            deltao = (targ - out).*oprime;                              
            deltah = hprime.*(woh(:,2:numhid+1)'*deltao);           
        
            %output weight update
            dwo= learnho.*(deltao*[h_layer]') + dwoldo*a;           
            woh = woh + dwo;                        
        
            %hidden weight update
            dwh = learnih.*((deltah)*in') + dwoldh*a;        
            whi = whi + dwh;                            

            %momentum memory
            dwoldh = dwh;   %for momentum                              
            dwoldo = dwo;   %for momentum                           
        
            %SSE
            trainingError = trainingError + 0.5*sum((targ-out).^2)/totalImages;
    
        end
    end
    
    [percentWrong, testE, b] = PPAFFATesterTwoLayer(whi, woh, testingData, activation);
    
    if(steps>numIter) %resets network if it reaches poor local minima
        %whi = (normrnd(0,1/sqrt(3),[numhid,inlen+1]));    %randomized weight matrix for input layer to hidden layer                 
        %woh = (normrnd(0,1/sqrt(3),[targlen,numhid+1]));  %randomized weight matrix for hidden layer to output layer
        %dwoldo = 0;
        %dwoldh = 0;
        %steps = 0;
        %attempts = attempts + 1;
        %Error = [];
        %testError = [];
        trainingError = 0;

    end
%% Code for plotting Error and displaying learned weights  
    Error = [Error, trainingError];
    TestPercentWrong = [TestPercentWrong, percentWrong]; %Percent Wrong, SSE
    TestError = [TestError, testE];
    
    if trainingError < .01
        fprintf('************************************************')
        fprintf('\nThe network learned weights:\n');
        whi
        woh
        
        plot(Error, 'b');hold;
        plot(TestPercentWrong, 'r');
        plot(TestError, 'g');
        legend('Training: SSE', 'Testing: Percent Wrong', 'Testing: SSE');
        
        fprintf('It took %i steps with a learning const of %g\n',steps,learnho);
        fprintf('It took %i atempts', attempts);
        fprintf('\nNOTE: First column of each weight matrix contains\n the bias weights\n');
        
        fprintf('************************************************\n')
        break
    end   
end