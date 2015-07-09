function Filter_sum = generate_angles(phi,theta, width)
% generate two lines spanning a particular angle, rotated at a given angle.
% phi, the angle between two lines
% theta, the rotation of the image
% usage: Filter_sum = generate_angles(pi/6,pi/20,'width',100); imagesc(Filter_sum); axis square

x_seq = linspace(-1,1,width);
y_seq = linspace(-1,1,width);

% f = sin(f*x_seq)
[X_mesh,Y_mesh] = meshgrid(x_seq,y_seq);
x_vec = X_mesh(:);
y_vec = Y_mesh(:);

% this function is a line between y = 0 to -1
xlim = 2/(width-1);
f = @(x,y) ( x<=xlim & x>=-xlim & y<=0 & y>=-1);
z_vec = f(x_vec,y_vec);
Z = reshape(z_vec, [width, width]);

     

Filter1 = imrotate(Z,-theta/pi*180,'bilinear','crop');
Filter2 = imrotate(Z,-(theta+phi)/pi*180,'bilinear','crop');


Filter_sum = (Filter1+Filter2)/2;

