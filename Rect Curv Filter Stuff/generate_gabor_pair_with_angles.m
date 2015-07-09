function Gabor_sum = generate_gabor_pair_with_angles(phi,theta,varargin)

% create two gabor fitlers with an angle of phi
width = 51;
f = 30;

if nargin >2 
    for i = 1:(nargin-2)/2
        varname = varargin{2*i-1};
        varval = varargin{2*i};
        switch varname
            case 'f'
                f = varval;
            case 'width'
                width = round(varval/2)*2+1;
            otherwise
        end
    end
end

width0 = 200;
x_seq = linspace(-2,2,width0);
y_seq = linspace(-2,2,width0);

% f = sin(f*x_seq)
[X_mesh,Y_mesh] = meshgrid(x_seq,y_seq);
x_vec = X_mesh(:);
y_vec = Y_mesh(:);


R1 = [cos(theta), -sin(theta); sin(theta), cos(theta)];
W = [x_vec,y_vec];
New_W = W*R1;
z_vec = cos(f*New_W(:,1));
Sigma = [0.02,0; 0,1.1]*0.1;
Mu = [0,-0.6];
h_Gaussian = mvnpdf(W,Mu*R1',R1*Sigma*R1');
z_vec = z_vec.*h_Gaussian;
Z = reshape(z_vec, [width0, width0]);
Gabor1 = Z;

% rotate the image
% rotation matrix R

R2 = [cos(phi+theta), -sin(phi+theta); sin(phi+theta), cos(phi+theta)];
New_W2 = W*R2;
z_vec = cos(f*New_W2(:,1));
h_Gaussian = mvnpdf(W,Mu*R2',R2*Sigma*R2');
z_vec = z_vec.*h_Gaussian;
Z = reshape(z_vec, [width0, width0]);
Gabor2 = Z;


%figure; imagesc(Gabor1+Gabor2);

% imrotate does not work well, it introduces small difference in different
% filterls. 
%Gabor2 = imrotate(Gabor1,phi/(2*pi)*360, 'bilinear','crop');

Gabor_sum = (Gabor1+Gabor2)/2;
Gabor_sum = imresize(Gabor_sum,[width,width],'Antialiasing',true);
%Gabor_sum = Gabor_sum./sum(abs(Gabor_sum(:)));


% debug, should be removed
% figure; 
% subplot(131)
% imagesc(Gabor1); colormap('gray'); axis square;
% subplot(132);
% imagesc(Gabor2); colormap('gray'); axis square;
% subplot(133);
% imagesc(Gabor_sum); axis square;
