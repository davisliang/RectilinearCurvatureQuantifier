function Dictionary = generate_angled_gabor_dictionary(phi,theta, varargin)

% phi, the angle between two gabor patches
%   range [0,pi]
% theta, the rotational angle 
%   range [0,2*pi]


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

n_phi = length(phi);
n_theta = length(theta);
Dictionary = cell(n_phi,n_theta,1);

for i = 1:n_phi
    for j = 1:n_theta
        Z = generate_gabor_pair_with_angles(phi(i), theta(j), 'f',f,'width',width);
        Dictionary{i,j} = Z;
    end
end
