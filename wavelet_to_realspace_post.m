function [X,Y,ZZ,time,hash_table,E,phi] = wavelet_to_realspace_post(filename,max_level)
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
        asgdata(d).dx   = (asgdata(d).max-asgdata(d).min)/2.0^double(max_level);
        asgdata(d).vec  = asgdata(d).min:asgdata(d).dx:asgdata(d).max;
        asgdata(d).FMWT = OperatorTwoScale_wavelet2(double(deg),max_level);
    end

%     fprintf('---- Solution Data ----------------------------\n');
%     fprintf('-- time = %5.4f\n',time);
%     fprintf('-- degree = %d\n',deg);
%     fprintf('-- dof = %5d\n',dof)
%     fprintf('-- max-level = %4d\n',max_level);
%     fprintf('-- dim0 --\n');
%     fprintf(' [min,max] = [%3.2f,%3.2f]\n',x_min,x_max)
%     fprintf(' lev = %2d\n',lev_x);
%     fprintf('-- dim1 --\n');
%     fprintf(' [min,max] = [%3.2f,%3.2f]\n',y_min,y_max)
%     fprintf(' lev = %2d\n',lev_y);
    
    %Get eletric field
    E = h5read(filename,'/Efield');
    phi = h5read(filename,'/phi');


    element_mat = reshape(h5read(filename,'/elements'),2*ndims,[]);
    for d=1:ndims
        asgdata(d).coords_lev = element_mat(d,:);
        asgdata(d).coords_pos = element_mat(d+ndims,:);
        asgdata(d).index      = lev_cell_to_1D_index(asgdata(d).coords_lev,asgdata(d).coords_pos);
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
    wave_tensor = tensor(@zeros,ones(1,ndims)*double(2^max_level*deg));

    dim_idx = cell(1,ndims);
    for i=1:active
        for d=1:ndims
            %Reverse order is due to reverse kron storage
            dp = ndims-d+1;
            dim_idx{d} = (asgdata(dp).index(i)-1)*double(deg)+1:asgdata(dp).index(i)*double(deg);
        end
        wave_tensor(dim_idx{:}) = reshape(soln((i-1)*ele_dof+1:i*ele_dof),ones(1,ndims)*double(deg));
    end

    % Translate from wavelets to realspace
    FT = {asgdata(:).FMWT};
    for d=1:ndims; FT{d} = FT{d}'; end
    real_tensor = ttm(wave_tensor,FT,1:ndims);



    %Legendre polynomial values for modal to nodal transformation
    [q,~] = lgwt(double(deg),-1,1); q = q';
    L = zeros(deg,deg);
    for i=1:deg
        tmp = legendre(i-1,q,'norm');
        L(i,:) = tmp(1,:);
    end

    dLcell = cell(1,ndims);
    bDiag = cell(1,2^max_level);
    for d=1:ndims
        asgdata(d).nodes = zeros(deg*2^max_level,1);
        for i=1:2^max_level
            %Create nodes
            asgdata(d).nodes(deg*(i-1)+1:deg*i) = ...
                asgdata(d).dx/2*q + (asgdata(d).vec(i)+asgdata(d).vec(i))/2;
            %Create scaled translation from modal to nodal data
            bDiag{i} = L'*sqrt(2/asgdata(d).dx);
        end
        dL = matlab.internal.math.blkdiag(bDiag{:});
        dLcell{d} = dL;
    end

    vals = ttm(real_tensor,dLcell,1:ndims);
    ZZ = double(vals);

    if ndims == 2
        [XX,YY] = meshgrid(asgdata(1).nodes,asgdata(2).nodes);
        %%Plot
        h = surf(XX,YY,ZZ,'EdgeColor','none');
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
