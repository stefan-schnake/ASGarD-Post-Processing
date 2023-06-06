function [nodes,real_tensor,time,hash_table,asgdata] = wavelet_to_realspace_post(filename,max_level)
    

    % Check for tensor toolbox
    assert(exist('tensor_toolbox-v3.5', 'dir'),"Tensor toolbox ..." + ...
        "not found. Download tensor toolbox at v3.5 at ..." + ...
        "https://gitlab.com/tensors/tensor_toolbox/-/releases/v3.5");
    addpath('tensor_toolbox-v3.5/');

    ndims = h5read(filename,'/ndims');
    deg = h5read(filename,'/degree');
    if nargin == 1
        max_level = h5read(filename,'/max_level');
    end
    time = h5read(filename,'/time');
    dof = h5read(filename,'/dof');

    for d=1:ndims
        asgdata(d).min  = h5read(filename,sprintf('/dim%1d_min',d-1)); 
        asgdata(d).max  = h5read(filename,sprintf('/dim%1d_max',d-1)); 
        asgdata(d).lev  = h5read(filename,sprintf('/dim%1d_level',d-1));
        assert(asgdata(d).lev <= max_level);
        asgdata(d).dx   = (asgdata(d).max-asgdata(d).min)/2.0^double(asgdata(d).lev);
        asgdata(d).vec  = asgdata(d).min:asgdata(d).dx:asgdata(d).max;
        asgdata(d).FMWT = OperatorTwoScale_wavelet2(double(deg),asgdata(d).lev);
    end

    fprintf('---- Solution Data ----------------------------\n');
    fprintf('-- time = %5.4f\n',time);
    fprintf('-- degree = %d\n',deg);
    fprintf('-- dof = %5d\n',dof)
    fprintf('-- max-level = %4d\n',max_level);
    for d=1:ndims
        fprintf('-- dim%1d --\n',d);
        fprintf(' [min,max] = [%3.2f,%3.2f]\n',asgdata(d).min,asgdata(d).max)
        fprintf(' lev = %2d\n',asgdata(d).lev);
        fprintf(' dx = %3.2f\n',asgdata(d).dx);
    end
    
    %Get eletric field
    %E = h5read(filename,'/Efield');
    %phi = h5read(filename,'/phi');


    element_mat = reshape(h5read(filename,'/elements'),2*ndims,[]);
    for d=1:ndims
        asgdata(d).coords_lev = element_mat(d,:);
        asgdata(d).coords_pos = element_mat(d+ndims,:);
        asgdata(d).index      = lev_cell_to_1D_index(asgdata(d).coords_lev,asgdata(d).coords_pos)';
    end

    active = numel(asgdata(1).coords_lev);
    idx = asgdata(ndims).index;
    for d=ndims-1:-1:1
        idx = idx + (asgdata(d).index-1)*double(2^((ndims-d)*max_level));
    end
    ele_dof = deg^ndims;
    
    soln = h5read(filename,'/soln');

    %Hashtable creation
    hash_table.elements_idx = zeros(1,active);
    hash_table.elements.lev_p1 = sparse([],[],[],2^(ndims*max_level),ndims);
    hash_table.elements.pos_p1 = sparse([],[],[],2^(ndims*max_level),ndims);
    hash_table.elements_idx = idx;
    for d=1:ndims
        hash_table.elements.lev_p1(idx,d) = asgdata(d).coords_lev'+1;
        hash_table.elements.pos_p1(idx,d) = asgdata(d).coords_pos'+1;
    end

    %Tensor time
    real_tensor = tensor(@zeros,2.^([asgdata(:).lev])*deg);

    perm_mat = zeros(deg^ndims,ndims);
    perm_vec = zeros(deg^ndims,1);
    for i=1:deg^ndims
        val = i-1;
        for d=1:ndims
            mod_val = mod(val,deg);
            perm_mat(i,d) = mod_val + 1;
            val = (val-mod_val)/deg;
        end
        for d=1:ndims
            perm_vec(i) = perm_vec(i) + (perm_mat(i,d)-1)*deg^(ndims-d);
        end
        perm_vec(i) = perm_vec(i)+1;
    end

    dim_idx = cell(1,ndims);
    for i=1:active
        for d=1:ndims
            %Reverse order is due to reverse kron storage
            %dp = ndims-d+1;
            %dim_idx{d} = (asgdata(dp).index(i)-1)*double(deg)+1:asgdata(dp).index(i)*double(deg);
            dim_idx{d} = (asgdata(d).index(i)-1)*double(deg)+1:asgdata(d).index(i)*double(deg);
            
        end
        %disp(dim_idx);
        tmp = soln((i-1)*ele_dof+1:i*ele_dof);
        real_tensor(dim_idx{:}) = reshape(tmp(perm_vec),ones(1,ndims)*double(deg));
    end

    % Translate from wavelets to realspace
    FT = {asgdata(:).FMWT};
    for d=1:ndims; FT{d} = FT{d}'; end
    real_tensor = ttm(real_tensor,FT,1:ndims);


    %Legendre polynomial values for modal to nodal transformation
    [q,w] = lgwt(double(deg),-1,1); q = q';
    L = zeros(deg,deg);
    for i=1:deg
        tmp = legendre(i-1,q,'norm');
        L(i,:) = tmp(1,:);
    end

    dLcell = cell(1,ndims);
    for d=1:ndims
        bDiag = cell(1,2^asgdata(d).lev);
        asgdata(d).nodes = zeros(deg*2^asgdata(d).lev,1);
        asgdata(d).weights = zeros(deg*2^asgdata(d).lev,1);
        for i=1:2^asgdata(d).lev
            %Create nodes
            asgdata(d).nodes(deg*(i-1)+1:deg*i) = ...
                asgdata(d).dx/2*q + (asgdata(d).vec(i)+asgdata(d).vec(i+1))/2;
            %Create weights
            asgdata(d).weights(deg*(i-1)+1:deg*i) = w*asgdata(d).dx/2;
            %Create scaled translation from modal to nodal data
            bDiag{i} = L'*sqrt(2/asgdata(d).dx);
        end
        %Create block diagonal matrix
        dL = matlab.internal.math.blkdiag(bDiag{:});
        dLcell{d} = dL;
    end

    real_tensor = ttm(real_tensor,dLcell,1:ndims);
    %ZZ = double(real_tensor);
    nodes = {asgdata(:).nodes};

    if ndims == 2
        %[XX,YY] = meshgrid(asgdata(1).nodes,asgdata(2).nodes);
        XX = double(ttensor(tensor(1,[1 1]),{asgdata(1).nodes,ones(numel(asgdata(2).nodes),1)}));
        YY = double(ttensor(tensor(1,[1 1]),{ones(numel(asgdata(1).nodes),1),asgdata(1).nodes}));
        %%Plot
        h = surf(XX,YY,real_tensor.data,'EdgeColor','none');
        view([0 90]); 
        %caxis([-0.05 0.35]);
        %zlim([-0.05 0.35]); 
        xlim([asgdata(1).min,asgdata(1).max]); ylim([asgdata(2).min,asgdata(2).max]); colorbar;
        %title(sprintf('time = %5.4f, dof = %d, maxlev = %d, lev = [%d,%d], thresh = %2.1e',time,dof,max_level,lev_x,lev_y,1e-6));
        coord = get_my_realspace_coord([asgdata(:).min],[asgdata(:).max],hash_table);
        z_max = max(max(get(h,'ZData')));
        hold on
        line(coord(:,1),coord(:,2),0*coord(:,1)+z_max,'Color','k','LineStyle',"none",'Marker',".",'MarkerSize',4);
        hold off
    end


end


function index = lev_cell_to_1D_index(lev,cell)

%%
% Map lev,cell to index within a single dim

index= 2.^(double(lev)-1)+double(cell)+1;

ix = find(lev == 0);
index(ix)=1;

end
