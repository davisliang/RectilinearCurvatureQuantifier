function Z = banana_filter(f,alpha,c,x_vec,y_vec, sigma_x, sigma_y)
% the mapping is from 2-D to a complex plane,
% just like Gabor filters, it has sine and cosine components. 

% f, frequency
% alpha, orientation of the gratings (rotation)
% c, curvature, C(x,y) = (x-c*y^2,y), 
%   this is the curvature at the center of the pattern. 
%   consider mapping x=0 using C(x,y), w = -c*u^2,  compute the curvature
%   \frac{y''}{{1+y'^2}^{3/2}}  of it, when u =0, it is c. 
% s, size of the pattern
% x_vec, y_vec, the range of x and y coordinates
%     to make sure the curvature's unit is 1/pixel, the step in x_vec and y_vec
% must be 1

% eg. Z = banana_filter(0.5,7/8*pi,0.09, -20:1:20, -20:1:20, 1, 0.7*pi);
% eg. imagesc(real(banana_filter(0.5,7/8*pi, 0.05, -20:1:20, -20:1:20, 0.4, 2*pi)))

% gamma is a constant
gamma = 0.08;

[X,Y] = meshgrid(x_vec, y_vec);
X_vec_long = X(:);
Y_vec_long = Y(:);


% stipr97
% rotation
M = [cos(alpha), sin(alpha); -sin(alpha), cos(alpha)];
M_x = [X_vec_long, Y_vec_long] * M';
x_c  = M_x(:,1) - c * M_x(:,2).^2;
y_c = M_x(:,2);

G = exp( -0.5*f.^2.*((x_c./sigma_x).^2 + (y_c./sigma_y).^2)); 
F = exp(1i*f.*x_c);

Z_vec_long =  gamma .*G.* (F); 
Z = reshape(Z_vec_long, length(x_vec), length(y_vec));

% figure; 
% imagesc(reshape(G,length(x_vec),length(x_vec)));
% colormap('gray');
% title('The curved gaussian patch');
% 
% figure;
% subplot(1,2,1); 
% imagesc(real(Z)); colormap('gray'); colorbar; axis square;
% title('Real part')
% subplot(1,2,2); 
% imagesc(imag(Z)); colormap('gray'); colorbar; axis square;
% title('Imaginary part')

