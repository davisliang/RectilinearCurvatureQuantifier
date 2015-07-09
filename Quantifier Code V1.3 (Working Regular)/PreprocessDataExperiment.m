function [] = PreprocessDataExperiment(imageLocation)
%Yufei's preprocessing code
close all; 
clc;
display(['Initiating Experiment Preprocessor...']);
s = 5; %scale
o = 8; %orientation
im_size = 225; %image size
size_gb=100; %gabor size
k = zeros(1, s); %attenuation factor
phi = zeros(1, o);  %orientation factor
std_dev=pi;
num_pca=8;
dwsp = 15;%downsampled rate
f_filtered_normalized_dwsp_vector_allsub_data = zeros(dwsp*dwsp*s*o, 1);
matPATH = 'experimentalData.mat';


matFileLoc = '/Users/davisliang/Desktop/'; %matFileLocation
matFile = 'PrepFaceVsScene_Tr90_Te8_RegularGabor.mat'; %matfile name
cd(matFileLoc);
matFile = load(matFile);
Mean = matFile.organizedData.MEAN;
standardDeviation = matFile.organizedData.STD;
cov_scale = matFile.organizedData.cov_scale;
mean_images = matFile.organizedData.mean_images;
total_train = matFile.organizedData.total_train;
mean_subst = matFile.organizedData.mean_subst;





%% Gabor filter
display(['Initializing Gabor Filters...']);
for i=1:5
    k(i)=(2*pi/size_gb)*2^i;
end
%orientation
for i=1:8
    phi(i)=(pi/8)*(i-1);
end
carrier = zeros(size_gb, size_gb);
envelop = zeros(size_gb, size_gb);
gabor = zeros(size_gb, size_gb, o, s);

for scale=1:size(k,2)
    %figure;
    for orientation=1:size(phi,2)
        for ii=-size_gb+1:size_gb
            for j=-size_gb+1:size_gb
                carrier(ii+size_gb,j+size_gb)=exp(1i*(k(scale)*cos(phi(orientation))*ii+k(scale)*sin(phi(orientation))*j));
                envelop(ii+size_gb,j+size_gb)=exp(-(k(scale)^2*(ii^2+j^2))/(2*std_dev*std_dev));
                gabor(ii+size_gb,j+size_gb,orientation,scale)=carrier(ii+size_gb,j+size_gb)*envelop(ii+size_gb,j+size_gb);
            end
        end
                           %subplot(2,4,orientation); imshow(gabor(:,:,orientation,scale),[]);
    end
end

%%


display(['Processing Test Image... Please Hold']);


%% for each image, do gabor_filtering
f=imread(imageLocation);
if size(size(f),2)==2
    f=imresize(im2double(f),[im_size im_size]);
else
    f=rgb2gray(im2double(f));
    f=imresize(f,[im_size im_size]);
end

%constructing gabor filter (16*16) and filtering the input image
for scale=1:size(k,2)
    %scale
    for orientation=1:size(phi,2)
        %subplot(2,4,orientation); imshow(gabor(:,:,orientation),[]);     
        f_filtered_dwsp{scale}(:,:,orientation)=imresize(imfilter(f,gabor(:,:,orientation,scale),'replicate','conv'),[dwsp,dwsp]);       
        %figure;
        %imshow(f_filtered{scale}(:,:,orientation),[])
    end
    %now we have 8 orientations for each scale, downsample and do normalization
    for orientation=1:size(phi,2)
        f_filtered_normalized_dwsp{scale}(:,:,orientation)=abs(f_filtered_dwsp{scale}(:,:,orientation));               
    end
end
%normalize them for each scale
for scale=1:size(k,2);
    f_filtered_normalized_dwsp_vector((scale-1)*dwsp*dwsp*size(phi,2)+1:(scale)*dwsp*dwsp*size(phi,2))=f_filtered_normalized_dwsp{scale}(:);             
end

f_filtered_normalized_dwsp_vector_allsub_data(:) = f_filtered_normalized_dwsp_vector(:); %will check correct sizing, verification

clear f_filtered_normalized_dwsp_vector;


%%%%%%%%%%%z-score%%%%%%%%%%%
f_filtered_normalized_dwsp_vector_allsub_data=bsxfun(@minus, f_filtered_normalized_dwsp_vector_allsub_data, Mean);
f_filtered_normalized_dwsp_vector_allsub_data=bsxfun(@rdivide, f_filtered_normalized_dwsp_vector_allsub_data, standardDeviation+.1);
%%%%%%%%%%%z-score%%%%%%%%%%%

%% PCA on different scale
    
    
    
    
for scales=1:size(k,2) 
    scale_all_data=f_filtered_normalized_dwsp_vector_allsub_data((scales-1)*dwsp*dwsp*size(phi,2)+1:scales*dwsp*dwsp*size(phi,2),:);
        
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
        
    f_PCA_temp_normal((scales-1)*num_pca+1:scales*num_pca,:)=f_PCA_scale_normal;

end

%% save data


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