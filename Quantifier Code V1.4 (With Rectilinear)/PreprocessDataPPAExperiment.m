function [] = PreprocessDataPPAExperiment(imageLocation)
%preprocessing Code
%Davis Liang, 6/16/15
%github: davisliang, email: d1liang@ucsd.edu
%Tarr Labs, CMU

%function [] = PreprocessDataSet(topFocus, fullFocus, loc, imshow,num_each, num_test, matPATH,fileType, numFolders)

%% intializing
close all;
display(['Initiating Rectilinear Experiment Preprocessor...']);

numScales = 3;
numOrientations = 8;
imageSize = 225;     %dwsp'ed image size?
angles = 8; %corresponds to 2pi/8 = pi/4 angle increments.. 0, 45, 90, 135, 180, 225, 270, 315, 360...
k = zeros(1,numScales);            % 0 0 0 0 ... number of scales (5) zeroes in a single row.
phi = zeros(1,numOrientations); 	% 0 0 0 0 0 ... number of orientations (8) zeroes in a single row.
num_pca = 8;    %number of principle components
dwsp = 15;       %downsampling rate, increased
gaborSize = 20;

%next set the path location of the files and upload the files.

filterType = 'rectil'; %rectilinear

numCurves = 8;
frequency = 0.3;
maxSize = 99;

f_filtered_normalized_dwsp_vector_allsub_data = zeros(imageSize*numScales*numOrientations*angles, 1);

matPATH = 'experimentalData.mat';

matFileLoc = '/Users/davisliang/Desktop/'; %matFileLocation
matFile = 'PrepFaceVsScene_Tr90_Te8.mat'; %matfile name
cd(matFileLoc);
matFile = load(matFile);
Mean = matFile.organizedData.MEAN;
standardDeviation = matFile.organizedData.STD;
cov_scale = matFile.organizedData.cov_scale;
mean_images = matFile.organizedData.mean_images;
total_train = matFile.organizedData.total_train;
mean_subst = matFile.organizedData.mean_subst;

%% generate rectilinear gabor filters
display(['Initializing Gabor Filters...']);
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

display(['Building Rectilinear Gabors...']);

if filterType == 'rectil'
    
    for j = 1:numScales
        Dictionary{j} = generate_angled_gabor_dictionary(phi,theta, 'width', gaborSize*(((pi/16)/sizeMod(j))), 'f', 40);
    end
    
elseif filterType == 'banana'
    
    Dictionary = generate_curved_gabor_dictionary(numCurves, numOrientations, numScales, frequency, maxSize);
    
else
    
    %nothing
    
end


%% Debugging code (plotting all gabor filters)
%counter = 1;
%for s = 1:numScales
%    figure;
%    for t = 1:8
%        for p = 1:8
%            if counter>64
%                counter = 1;
%            end
%            subplot(8,8,counter);
%            temp = Dictionary{s}{p,t};
%            imagesc(real(temp));
%            colormap('gray');
%            axis square;
%            counter=counter+1;
%        end
%    end
%end
%320 filters total
%Dictionary{size}{phi,theta} theta is the primary angle (from the y-axis, clockwise) and phi is the extension of the primary angle 

%% filtering
display(['Processing Test Image... Please Hold']);
f = imread(imageLocation);


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
    
f_filtered_normalized_dwsp_vector_allsub_data(:) = f_filtered_normalized_dwsp_vector(:); %will check correct sizing, verification
clear f_filtered_normalized_dwsp_vector

    
%% zscore (use training set mean and std to calculate zscore for the test set).


f_filtered_normalized_dwsp_vector_allsub_data = bsxfun(@minus, f_filtered_normalized_dwsp_vector_allsub_data, Mean);
f_filtered_normalized_dwsp_vector_allsub_data = bsxfun(@rdivide, f_filtered_normalized_dwsp_vector_allsub_data, standardDeviation + 0.1);
    
   


    %PCA on different scale
    
    
    
    
for s=1:size(k,2) 
    scale_all_data=f_filtered_normalized_dwsp_vector_allsub_data((s-1)*dwsp*dwsp*size(phi,2)*size(theta,2)+1:s*dwsp*dwsp*size(theta,2)*size(phi,2),:);

        
    %turk and pentland trick, for each scale

    mean_subst_data=scale_all_data-repmat(mean_images,1,1);
    
    [vector_temp, value]=eig(cov_scale);
    vector_biggest=vector_temp(:,end-num_pca+1:end);
        
    %original principal components
    vector_ori=mean_subst*vector_biggest;
    vector_ori = normc(vector_ori);
    %projection onto the basis vector vector_ori(dimension 512-dimension 8)

    
    %normal
    %%%%%z-score%%%%%
    temp = vector_ori'*mean_subst;
    s=std(temp,[],2);
    m=mean(temp,2);
    
    temp = vector_ori'*mean_subst_data;
    temp=bsxfun(@minus, temp, m);
    f_PCA_scale_normal=bsxfun(@rdivide, temp, s);
    %%%%%%%%%%%%%%%%%%%%

%         f_PCA_scale_normal=(vector_ori(:,:,scales)'*mean_subst(:,:,scales));
%         f_PCA_scale_test_normal=(vector_ori(:,:,scales)'*mean_subst_test(:,:,scales));        
        
    f_PCA_temp_normal((numScales-1)*num_pca+1:numScales*num_pca,:)=f_PCA_scale_normal;
end


%% save stuff
experimentData.data = f_PCA_temp_normal(:);


%a = strcat('pofa_zscore_PreprocessedData_identity_',int2str(num_pca),'.mat');
%save(a,'data')
save([strcat('/Users/davisliang/Desktop/',matPATH)],'experimentData')

%train_image = files(train_indice == 1);
%test_image = files(train_indice == 0);

%information.train= train_image;
%information.test = test_image;



%a = strcat('pofa_zscore_identity_info.mat');
%save(a,'information');

end