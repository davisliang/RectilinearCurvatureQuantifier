%preprocessing Code
%Davis Liang, 6/16/15
%github: davisliang, email: d1liang@ucsd.edu
%Tarr Labs, CMU

%function [] = PreprocessDataSet(topFocus, fullFocus, loc, imshow,num_each, num_test, matPATH,fileType, numFolders)

%% intializing
close all;
clear all;

numScales = 3;
numOrientations = 8;
imageSize = 64;     %dwsp'ed image size?
gaborSize = 48;     %max gabor size?
angles = 8; %corresponds to 2pi/8 = pi/4 angle increments.. 0, 45, 90, 135, 180, 225, 270, 315, 360...

k = zeros(1,numScales);            % 0 0 0 0 ... number of scales (5) zeroes in a single row.
phi = zeros(1,numOrientations); 	% 0 0 0 0 0 ... number of orientations (8) zeroes in a single row.

std_dev = 1;   %gabor's standard deviation
num_pca = 8;    %number of principle components
dwsp = 20;       %downsampling rate, increased

numTest = 4;
numTrain = 12;
numFolders = 5;

gaborFreq = 50;
gaborWidth = 200;

sdConst = 0.1;


%next set the path location of the files and upload the files.
root = '/Users/davisliang/Desktop/DataSets/NewIdentity/';
fileType = '.BMP';

filterType = 'banana'; %rectilinear

numCurves = 8;
frequency = 0.3;
maxSize = 99;

%read from files
files_temp = [dir(fullfile(root,strcat('*',fileType)))]; %file_temp now is an M by 1 structure that holds name, date, bytes, isdir, datenum of all files in the folder.
files = {}; %files is a CELL matrix. (can hold matrices of various sizes within)... AKA can hold any data.

%next, we get rid of all hidden and unrelated files
for i = 1:length(files_temp)
        if(files_temp(i).name(1) ~= '.')
            files = [files;files_temp(i).name]; %files is column cell vector of cell units of row vector arrays that hold names
        end
end

%add labels to the files
label = zeros(length(files),1); %initialize to a column of zeros
for i=1:length(files)   %iterating through the files
    label(i) = str2double(files{i}(1:2)); %adds the two numbers at the beginning of the filenames as the labels
end

[label, I] = sort(label);   %sorts labels, I corresponds to image number, allows for label sorting.
files = files(I);           %sorts files in case they are not in order. 

curr = 0;
prev = 0;

%this will simplify the labels to 1, 2, 3, ... etc.
for i = 1:length(label) %iterating through each image
        if(prev ~= label(i))    
            curr = curr+1;
            prev = label(i);
            label(i) = curr;
        else
            label(i) = curr;
        end
end

%following code tells us how many files are in each folder.
count = [1]; %initializes count as an element 1
for i = 2:length(label) %iterating through all the images except the first
    if label(i) ~= label(i-1) 
        count = [count,1];
    else
        count(end) = count(end)+1;
    end
end

%label the training and testing sets

%following code produces a vector that tells which images to train on
%(randomly chosen test set of predecided size).
train_indice = ones(length(files),1); 
start = 1;
for i = 1:length(count) 
    indice = randperm(count(i)); 
    for j = 1:numTest 
        train_indice(indice(j)+start-1)=0; 
    end
    start = start + count(i);
end

label_test = label(train_indice==0);    %folder labels for the test set
label_train = label(train_indice==1);   %folder labels for training set

total_train = sum(train_indice);                    %number of training images
total_test = length(train_indice) - total_train;    %number of testing images

%create the filtered image matrix of all zeroes. Includes an element for
%each pixel in the downsampled image, for each scales and orientation.
%64*5*8 
f_filtered_normalized_dwsp_vector_allsub_train = zeros(imageSize*numScales*numOrientations*angles, total_train);
f_filtered_normalized_dwsp_vector_allsub_test = zeros(imageSize*numScales*numOrientations*angles, total_test);

train_i = 1;
test_i = 1;

%% generate rectilinear gabor filters

%First, generate your angled filters according to size of image
%recall that phi is the rotation of the primary gabor whilst theta is the
%angle of rotation of the secondary gabor with respect to the first.

%iterate through size, iterate through orientation of first gabor, iterate
%through orientation of second gabor.
%have a set frequency and width

%what is the size of the gabor filters with respect to the size of the
%image (largest? Given I have 5 sizes for each orientation and angle, what
%is the attenuation factor for the gabors?)

%k corresponds to the size modulator constant (proportional to e^(-k))
for j=1:numScales
    sizeMod(j)=(2*pi/imageSize)*2^j;
end

%phi corresponds to the 8 orientation angles of the primary gabor
for j=1:8
    theta(j)=(pi/4)*(j-1);
end

%theta corresponds to the 8 orientation angles of the secondary gabor
for j=1:8
    phi(j)=(pi/8)*j;
end

if filterType == 'rectil'
    
    for j = 1:numScales
        Dictionary{j} = generate_angled_gabor_dictionary(phi,theta, 'width', 96*(((pi/16)/sizeMod(j))), 'f', 40);
    end
    
elseif filterType == 'banana'
    
    Dictionary = generate_curved_gabor_dictionary(numCurves, numOrientations, numScales, frequency, maxSize);
    
else
    
    %nothing
    
end


%% Debugging code (plotting all gabor filters)
counter = 1;
for s = 1:numScales
    figure;
    for t = 1:8
        for p = 1:8
            if counter>64
                counter = 1;
            end
            subplot(8,8,counter);
            temp = Dictionary{s}{p,t};
            imagesc(real(temp));
            colormap('gray');
            axis square;
            counter=counter+1;
        end
    end
end
%320 filters total
%Dictionary{size}{phi,theta} theta is the primary angle (from the y-axis, clockwise) and phi is the extension of the primary angle 

%% filtering
for i=1:length(files)
    display(['Image' num2str(i)]); %counts images
    file_path = strcat(root, files{i}); %creates file path from the cell array above



    %for each gabor, do filtering. See Yufei's code.
    f=imread(file_path);
    if size(size(f),2)==2
        f=imresize(im2double(f),[imageSize imageSize]);
    else
        f=rgb2gray(im2double(f));
        f=imresize(f,[imageSize imageSize]); %resizes image, probably with an average filter and a downsampling matrix.
    end
    %constructing gabor filter (16*16) and filtering the input image
    for scale = 1:size(sizeMod,2)
        for secO = 1:size(theta,2)
            for primO = 1:size(phi,2)
                f_filtered_dwsp{scale}(:,:,primO,secO) = imresize(imfilter(f,Dictionary{scale}{secO,primO},'replicate','conv'),[dwsp,dwsp]);
            end
        end
        
        for primO=1:size(theta,2)
            for secO=1:size(phi,2)
                f_filtered_normalized_dwsp{scale}(:,:,primO,secO) = abs(f_filtered_dwsp{scale}(:, :, primO, secO));
            end
        end
        
    end
    
    for scale=1:size(sizeMod,2)
        f_filtered_normalized_dwsp_vector((scale-1)*dwsp*dwsp*size(phi,2)*size(theta,2)+1:(scale)*dwsp*dwsp*size(theta,2)*size(phi,2)) = f_filtered_normalized_dwsp{scale}(:);
    end
    

    %assemble the test and the training set.
    if train_indice(i) == 0
        f_filtered_normalized_dwsp_vector_test(:,test_i) = f_filtered_normalized_dwsp_vector(:);
        test_i = test_i + 1;
    else
        f_filtered_normalized_dwsp_vector_train(:,train_i) = f_filtered_normalized_dwsp_vector(:);
        train_i = train_i + 1;
    end
    clear f_filtered_normalized_dwsp_vector
end
    
%% zscore (use training set mean and std to calculate zscore for the test set).
sd = std(f_filtered_normalized_dwsp_vector_train,[],2);
m = mean(f_filtered_normalized_dwsp_vector_train,2);
f_filtered_normalized_vector_dwsp_allsub_train = bsxfun(@minus, f_filtered_normalized_dwsp_vector_train, m);
f_filtered_normalized_vector_dwsp_allsub_train = bsxfun(@rdivide, f_filtered_normalized_dwsp_vector_train, s + 0.1);
f_filtered_normalized_vector_dwsp_allsub_test = bsxfun(@minus, f_filtered_normalized_dwsp_vector_test, m);
f_filtered_normalized_vector_dwsp_allsub_test = bsxfun(@rdivide, f_filtered_normalized_dwsp_vector_test, s + 0.1);
    
    


data.train = f_filtered_normalized_dwsp_vector_allsub_train;
data.test = f_filtered_normalized_dwsp_vector_allsub_test;
data.train_label = label_train;
data.test_label = label_test;
a = 'pofa_zscore_GaborData_identity.mat';
save(a,'data');

    %PCA on different scale
    
    
    
    
for s=1:size(k,2) 
    scale_all=f_filtered_normalized_dwsp_vector_allsub_train((s-1)*dwsp*dwsp*size(phi,2)*size(theta,2)+1:s*dwsp*dwsp*size(theta,2)*size(phi,2),:);
    scale_all_test=f_filtered_normalized_dwsp_vector_allsub_test((s-1)*dwsp*dwsp*size(phi,2)*size(theta,2)+1:s*dwsp*dwsp*size(theta,2)*size(phi,2),:);
    mean_images=mean(scale_all,2);    
        
    %turk and pentland trick, for each scale
    mean_subst=scale_all-repmat(mean_images,1,total_train);
    mean_subst_test=scale_all_test-repmat(mean_images,1,total_test);
    cov_scale=(mean_subst'*mean_subst)*(1/total_train); %(estimate of covariance)
    [vector_temp, value]=eig(cov_scale);
    vector_biggest=vector_temp(:,end-num_pca+1:end);
        
    %original principal components
    vector_ori=mean_subst*vector_biggest;
    vector_ori = normc(vector_ori);
    %projection onto the basis vector vector_ori(dimension 512-dimension 8)

    
    %normal
    %%%%%z-score%%%%%
    temp = vector_ori'*mean_subst;
    sd=std(temp,[],2);
    m=mean(temp,2);
    temp=bsxfun(@minus, temp, m);
    f_PCA_scale_normal=bsxfun(@rdivide, temp, sd);
    temp = vector_ori'*mean_subst_test;
    temp=bsxfun(@minus, temp, m);
    f_PCA_scale_test_normal=bsxfun(@rdivide, temp, sd);
    %%%%%%%%%%%%%%%%%%%%

%         f_PCA_scale_normal=(vector_ori(:,:,scales)'*mean_subst(:,:,scales));
%         f_PCA_scale_test_normal=(vector_ori(:,:,scales)'*mean_subst_test(:,:,scales));        
        
    f_PCA_temp_normal((s-1)*num_pca+1:s*num_pca,:)=f_PCA_scale_normal;
    f_PCA_test_temp_normal((s-1)*num_pca+1:s*num_pca,:)=f_PCA_scale_test_normal;
end


data.train = f_PCA_temp_normal;
data.test = f_PCA_test_temp_normal;
data.train_label = label_train;
data.test_label = label_test;


%SETUP OrganizedDataSet
organizedData.train = [];
organizedData.test = [];
organizedData.trainLabel = [];
organizedData.testLabel = [];
counter = 0;
for folders=1:numFolders
    for images=1:numTrain
        organizedData.train{folders}(:,images) = data.train(:,images+(counter*12));
        organizedData.trainlabel{folders}(images) = data.train_label(images);
    end

    for images=1:numTest
        organizedData.test{folders}(:,images) = data.test(:,images+(counter*2));
        organizedData.testlabel{folders}(images) = data.test_label(images);
    end
    counter = counter + 1;
end


%a = strcat('pofa_zscore_PreprocessedData_identity_',int2str(num_pca),'.mat');
%save(a,'data')
matPATH = strcat('prep_Faces_Curved_Tr_12_Te_4', '.mat');
save([strcat('/Users/davisliang/Desktop/',matPATH)],'organizedData')


%train_image = files(train_indice == 1);
%test_image = files(train_indice == 0);

%information.train= train_image;
%information.test = test_image;



%a = strcat('pofa_zscore_identity_info.mat');
%save(a,'information');