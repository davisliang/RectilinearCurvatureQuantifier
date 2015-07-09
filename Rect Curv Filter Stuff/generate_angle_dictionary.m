function Dictionary = generate_angle_dictionary(Param1,theta,width)

% generate a dictionary with any filter generation function
% usage:
% Phi = [0:5]/6*pi;
% Theta = [0:15]/16*2*pi;
% Dictionary = generate_angle_dictionary(Phi,Theta, 101)

n_Param1 = length(Param1);
n_theta = length(theta);
Dictionary = cell(n_Param1,n_theta,1);

for i = 1:n_Param1
    for j = 1:n_theta
             Z = generate_angles(Param1(i), theta(j),width);
        Dictionary{i,j} = Z;
    end
end
