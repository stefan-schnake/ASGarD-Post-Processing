function [coords] = get_tensor_coords(asgdata)
%Create a cell of the coordinates in each direction
ndims = numel(asgdata);

coords = cell(1,ndims);

nodes = cell(1,ndims);
for d=1:ndims
    nodes{d} = ones(numel(asgdata(d).nodes),1);
end
for d=1:ndims
    tmp = nodes;
    tmp{d} = asgdata(d).nodes;
    coords{d} = double(ttensor(tensor(1,ones(1,ndims)),tmp));
end
end