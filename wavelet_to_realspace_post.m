function [X,Y,ZZ,time] = wavelet_to_realspace_post(filename,max_level)

  
    deg = h5read(filename,'/degree');
    if nargin == 1
        max_level = h5read(filename,'/max_level');
    end
    time = h5read(filename,'/time');

    % x-dim data
    x_min = h5read(filename,'/dim0_min'); 
    x_max = h5read(filename,'/dim0_max');
    lev_x = h5read(filename,'/dim0_level');
    dx = (x_max-x_min)/2.0^double(max_level);
    x = x_min:dx:x_max;
    [FMWT_x,~] = OperatorTwoScale_wavelet2(double(deg),max_level);
    % y-dim data
    y_min = h5read(filename,'/dim1_min');
    y_max = h5read(filename,'/dim1_max');
    lev_y = h5read(filename,'/dim1_level');
    dy = (y_max-y_min)/2.0^double(max_level);
    y = y_min:dy:y_max;
    [FMWT_y,~] = OperatorTwoScale_wavelet2(double(deg),max_level);

    assert(lev_x <= max_level);
    assert(lev_y <= max_level);

    fprintf('---- Solution Data ----------------------------\n');
    fprintf('-- time = %5.4f\n',time);
    fprintf('-- degree = %d\n',deg);
    fprintf('-- dof = %5d\n',h5read(filename,'/dof'))
    fprintf('-- max-level = %4d\n',max_level);
    fprintf('-- dim0 --\n');
    fprintf(' [min,max] = [%3.2f,%3.2f]\n',x_min,x_max)
    fprintf(' lev = %2d\n',lev_x);
    fprintf('-- dim1 --\n');
    fprintf(' [min,max] = [%3.2f,%3.2f]\n',y_min,y_max)
    fprintf(' lev = %2d\n',lev_y);
    


    element_mat = reshape(h5read(filename,'/elements'),4,[]);
    %element_mat = [1;0;0;0];
    dim0_coords_lev = element_mat(1,:);
    dim1_coords_lev = element_mat(2,:);
    dim0_coords_pos = element_mat(3,:);
    dim1_coords_pos = element_mat(4,:);

    active = numel(dim0_coords_lev);

    index_x = lev_cell_to_1D_index(dim0_coords_lev,dim0_coords_pos);
    index_y = lev_cell_to_1D_index(dim1_coords_lev,dim1_coords_pos);

    idx = (index_x-1)*2^double(max_level) + index_y;
    ele_dof = deg^2;
    
    soln = h5read(filename,'/soln');
    %soln = [1;0;0;0;0;0;0;0;0];

    %Convert to [wave_1x,wave_2x,..] \otimes [wave_1y,wave_2y,...]
    wave_space = zeros(2^(2*max_level)*ele_dof,1);
    for i=1:active
        for degx = 1:deg
            for degy = 1:deg
                %wave_space((idx(i)-1)*ele_dof+(degx-1)*deg+degy) = soln((i-1)*ele_dof+(degx-1)*deg+degy);
                wave_space(((index_x(i)-1)*deg+degx-1)*2^max_level*deg+(index_y(i)-1)*deg+degy)= soln((i-1)*ele_dof+(degx-1)*deg+degy);
                %wave_space((idx(i)-1)*ele_dof+1:idx(i)*ele_dof) = soln((i-1)*ele_dof+1:i*ele_dof);
            end
        end
    end
    %wave_space = zeros(size(wave_space)); wave_space(2^max_level*deg+1) = 1;
    %Convert from wavelet space to realspace
    real_space = reshape(FMWT_y'*reshape(wave_space,[],2^max_level*deg)*FMWT_x,[],1);
    real_space_cell = zeros(size(real_space));
    %Convert to cell ordered vector
    count = 1;
    for i=1:2^max_level
        for j=1:2^max_level
            for degx=1:deg
                for degy=1:deg
                    %Create appropriate index
                    idx_i = (i-1)*deg+degx;
                    idx_j = (j-1)*deg+degy;
                    real_space_cell(count) = real_space((idx_i-1)*2^max_level*deg+idx_j);
                    count = count + 1;
                end
            end
        end
    end

    %Legendre polynomial values
    [q,~] = lgwt(double(deg),-1,1); q = q';
    L = zeros(deg,deg);
    for i=1:deg
        tmp = legendre(i-1,q,'norm');
        L(i,:) = tmp(1,:)*sqrt(2/dx);
    end

    X = zeros(deg*2^max_level,1);
    for i=1:2^max_level
        X(deg*(i-1)+1:deg*i) = dx/2*q + (x(i)+y(i+1))/2;
    end
    Y = zeros(deg*2^max_level,1);
    for j=1:2^max_level
        Y(deg*(j-1)+1:deg*j) = dy/2*q + (y(j)+y(j+1))/2;
    end
    [XX,YY] = meshgrid(X,Y);
    ZZ = zeros(numel(X),numel(Y));
    for i=1:2^max_level
        for j=1:2^max_level
            tmp = real_space_cell(ele_dof*((i-1)*2^max_level+j-1)+1:ele_dof*((i-1)*2^max_level+j));
            Z = L'*reshape(tmp,deg,deg)*L;
            ZZ((j-1)*deg+1:j*deg,(i-1)*deg+1:i*deg) = Z;
        end
    end


    %%Plot
    surf(XX,YY,ZZ,'EdgeColor','interp'); view([0 90]);
    xlim([x_min,x_max]); ylim([y_min,y_max]); colorbar;
    title(sprintf('time = %5.4f',time));


end


function index = lev_cell_to_1D_index(lev,cell)

%%
% Map lev,cell to index within a single dim

index= 2.^(double(lev)-1)+double(cell)+1;

ix = find(lev == 0);
index(ix)=1;

end
