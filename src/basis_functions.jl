# an element of a partition with coordinate center represented by PT
struct Domain{PT<:AbstractVector}
    # Centroid of partition. 
    center::PT 

    # Number of elementary basis functions it contains, needed to compute the new centroids
    n_descendants::Int 

    # The list of the children on the next finer scale.
    children::Vector{Domain{PT}}

    # an id to keep track of the partition elements
    id::Tuple{Int, Int}
end

function n_descendants(t::Domain)
    return t.n_descendants
end

function children(t::Domain)
    return t.children
end

function center(t::Domain)
    return t.center
end

function id(t::Domain)
    return t.id 
end

function descendants(t::Domain{PT}) where PT<:AbstractVector
    if iselementary(t::Domain)
        return [center(t)], [id(t)]
    else 
        out_center = Vector{PT}(undef, 0)
        out_id = Vector{Tuple{Int,Int}}(undef, 0)
        for ch in children(t)
            out = descendants(ch)
            append!(out_center, out[1])
            append!(out_id, out[2])
        end
        return out_center, out_id
    end
end

# construct a new domain from a list of domain on a finer scale
function Domain(input_children::AbstractVector{Domain{PT}}, id) where {PT<:AbstractVector}
    out_center = zero(PT)
    out_n_descendants = 0
    out_children = copy(input_children)
    for t in input_children
        out_n_descendants += n_descendants(t)
        out_center += center(t) * n_descendants(t)
    end
    out_center /= out_n_descendants
    return Domain{PT}(out_center, out_n_descendants, out_children, id)
end

# construct an elementary domain from a coordinate
function Domain(input_coordinates::PT, id) where {PT<:AbstractVector}
    return Domain{PT}(input_coordinates, 1, Vector{Domain{PT}}(undef, 0), id)
end

# We define a domain as elementary if it does not have any children. Note that this is slightly different than demanding it to consist of a single coordinate domain
function iselementary(t::Domain)
    return isempty(t.children)
end

function find_ranges(a::AbstractVector{<:Integer})
    @assert issorted(a); @assert a[1] == one(eltype(a))
    n = a[end]
    indices = Vector{eltype(a)}(undef, n + 1)
    track_el = zero(eltype(a))
    for (k, el) in enumerate(a)
        if el > track_el
            @assert el == track_el + 1
            # Note first index of size el
            indices[el] = k
            track_el = el
        end
    end
    indices[end] = length(a) + 1
    return map((x,y) -> x : (y - 1), indices[1 : (end - 1)], indices[2:end])
end

function haar_aggregation2d(coords, q)
    partitions = Vector{Vector{Domain{eltype(coords)}}}(undef, q)
    partitions[q] = Domain.(coords, [(q, k) for k = 1 : length(coords)])
    scales = 1 ./ 2 .^ (1 : q)
    for k = (q - 1) : -1 : 1
       nk = 2^k 
        hk = 1/(2^k)
        Nk = 4^k 
        linear = LinearIndices((1:nk, 1:nk))
        partitions[k] = Vector{Domain{eltype(coords)}}(undef, Nk)
        assignments = Vector{Int}(undef, length(partitions[k + 1]))
        for (i, domain) in enumerate(partitions[k + 1])
            c = center(domain)
            # The assignments of fine scale domains to different coarse scale domains
            assignments[i] = linear[Int(1 + div(c[1], hk)), Int(1 + div(c[2], hk))]
        end
        P = sortperm(assignments)
        ranges = find_ranges(assignments[P]) 
        @assert length(ranges) == length(partitions[k])
        for (i, r) in enumerate(ranges)
            partitions[k][i] = Domain(partitions[k + 1][P[r]], (k, i))
        end
    end
    return partitions, scales
end

# A function that uses a set of representative points for the dofs to compute the multiresolution basis
function geometric_aggregation(coords, h_min, h_max, h=1/2, tree_function=KDTree)
    @assert h_min < h_max
    # Determine number of levels
    q = round(Int, log(h, h_min / h_max))
    scales = h_min./ (h).^(q : -1 : 1)
    partitions = Vector{Vector{Domain{eltype(coords)}}}(undef, q)
    # creating the partition on the finest level
    partitions[q] = Domain.(coords, [(q, k) for k = 1 : length(coords)])

    # determine nested partition
    for k = (q-1) : -1 : 1
        # construct a tree consisting of the centers of the next finer domain
        centers = center.(partitions[k + 1])
        all_tree = tree_function(centers)

        # array used to keep track of nodes that have already been crossed off
        nodes_taken = falses(length(partitions[k+1]))
        # array of the centers used to perform the clustering
        aggregation_centers = Vector{eltype(coords)}(undef,0)

        # Find the neighborhoods of points that cannot jointly be included as centers
        neighborhoods = inrange(all_tree, center.(partitions[k+1]), scales[k + 1])

        # Finding the center indices
        for (center_index, nhd) in enumerate(neighborhoods)
            # Check if the member node has not been taken already
            if !(nodes_taken[center_index])
                # add the center to the aggretation centers
                push!(aggregation_centers, centers[center_index])
                # remove all nodes blocked by the new aggregation center from the list
                nodes_taken[nhd] .= true
            end
        end

        # Construct a tree only from the aggregation centers
        aggregation_tree = tree_function(aggregation_centers)
        # For each center on the next finer level, identify the aggregation center that is closest to it
        aggregation_list = nn(aggregation_tree, centers)[1]

        # rearrange the aggregation list into a list of Tupels, such that the first entry denotes the id of the new partition and the second index denotes the index of the child partition that is being added to it
        aggregation_list = map( x -> (x[2], x[1]), collect(enumerate(aggregation_list))); sort!(aggregation_list)

        # index ranges contains Unit ranges that, when indexing into aggregation_list, pick out the partitions assigned to the same parent
        index_ranges = Vector{UnitRange{Int}}(undef, length(unique(getindex.(aggregation_list, 1))))
        last_end = 0
        for k = 1 : (length(index_ranges) - 1)
            new_end = findnext(x -> x[1] != aggregation_list[last_end + 1][1], aggregation_list, last_end + 1) - 1
            index_ranges[k] = (last_end + 1) : new_end
            last_end = new_end
        end
        index_ranges[end] = (last_end + 1) : length(aggregation_list)

        # Initialize partition 
        partitions[k] = Vector{Domain}(undef, length(index_ranges))
        # For each index range associated to a domain on the k-th level we assemble indices of the children domains and fetch the associated domains. We then call the constructor on the resulting array of domains in order to compute the domain on the next coarser level.
        for l = 1 : length(index_ranges)
            partitions[k][l] = Domain(partitions[k+1][getindex.(aggregation_list[index_ranges[l]], 2)], (k, l))
        end
    end

    return partitions, scales
end

function geometric_multibasis(partitions::AbstractVector{<:AbstractVector{Domain{PT}}}) where PT<:AbstractVector
    q = length(partitions)
    N = length(partitions[q])
    # Storing the centers of the basis function
    centers = Vector{Vector{PT}}(undef, q)
    for k = 1 : q
        centers[k] = Vector{PT}(undef, 0)
    end
    # The parts of the eventual sparse matrix, in order to allow to easily append, which requires changing the length of the copltr vectors, we do not form an actual SparseMatrixCSC yet
    rowvals = Vector{Vector{Int}}(undef,q)
    colptrs = Vector{Vector{Int}}(undef,q)
    nzvals = Vector{Vector{Float64}}(undef,q)
    for k = 1 : q
        rowvals[k] = Vector{Int}(undef, 0)
        colptrs[k] = ones(Int, 1)
        nzvals[k] = Vector{Float64}(undef, 0)
    end

    # recursive function that gathers the domains
    function gather_𝐰(t::Domain{PT}) where PT<:AbstractVector
        # if we arrived at an elementary node, just return the id of that node
        if iselementary(t)
            return [id(t)[2]]
        else 
            # recursively apply the function to each child and gather the resulting descendant nodes. 
            index_list = [gather_𝐰(ch) for ch in children(t)] 
            m = length(index_list)
            master_list = reduce(vcat, index_list)         
            n = length(master_list)
            # temporary matrix corresponding to the basis set on the restriction to t
            # We are adding a bottom zero column to ensure that M is never short and fat, even when m = n, corresponding to the case where each child of t is an elementary basis function
            @assert n + 1 >= m + 1
            M = zeros(n + 1, m + 1)
            M[:, 1] .= 1
            offset = 0
            for k = 2 : m + 1
                M[(offset + 1) : (offset + length(index_list[k - 1])), k] .= 1 
                offset += length(index_list[k - 1])
            end
            M[end, :] .= 0.0
            # QR factorizing it to obtain orthonormal columns and appending to corresponding 𝐖 matrix.
            # We are removing the leading column (which is a constant vector) and the last row (which was added to ensure that M can never be "short and fat")
            Q = Matrix(qr(M).Q)[1 : (end - 1), 2 : (end - 1)]
            # Sparse matrix format assumes sorted nonzeros
            P = sortperm(master_list); Q = Q[P, :]; master_list = master_list[P]
            # appending to the sparse matrix representing the basis functions
            append!(rowvals[id(t)[1] + 1], repeat(master_list, m - 1))
            append!(nzvals[id(t)[1] + 1], vec(Q))
            append!(colptrs[id(t)[1] + 1], colptrs[id(t)[1] + 1][end] .+ Vector(1 : (m - 1)) * n)
            append!(centers[id(t)[1] + 1], repeat([center(t)], m - 1))
            return master_list
        end
    end
    # apply recursion to all top level
    for t in partitions[1]
        master_list = gather_𝐰(t)
        append!(rowvals[1], master_list) 
        append!(nzvals[1], ones(length(master_list)) / sqrt(length(master_list))) 
        append!(colptrs[1], colptrs[1][end] .+ [1] * length(master_list))
        append!(centers[1], [center(t)])
    end

    𝐖s = Vector{SparseMatrixCSC{Float64,Int}}(undef, q)
    𝐖s[1] = SparseMatrixCSC(N, length(partitions[1]), colptrs[1], rowvals[1], nzvals[1])
    for k = 2 : q
        𝐖s[k] = SparseMatrixCSC(N, length(partitions[k]) - length(partitions[k-1]), colptrs[k], rowvals[k], nzvals[k])
    end        

    return 𝐖s, centers
end

# Function that creates the coloring and sparsity pattern.
function geometric_coloring(scales, 𝐖s, centers, ρ, tree_function=KDTree)
    q = length(𝐖s)
    N = sum(size.(𝐖s, 2))
    # Vector that contains the colors as unit ranges.
    colors = Vector{Vector{UnitRange{Int}}}(undef, q)
    # Vector that contains the multicolor ordering
    Ps = Vector{Vector{Int}}(undef, q)
    # The Sparse matrix that will contain the sparsity pattern
    # We construct the forval and colptr vectors. Note that when constructed, we index the rowwval in the original ordering and the columns in the multicolor ordering.
    rowval = Vector{Int}(undef, 0)
    colptr = ones(Int, 1)
    # Vector containing all centers of basis functions, which is used to index the entries of rowval
    all_centers = reduce(vcat, centers)
    for k = 1 : q
        # keep track how many nodes have been colored
        colored = 0
        Nk = size(𝐖s[k], 2)
        nodes_colored = falses(Nk)
        Ps[k] = Vector{Int}(undef, 0)
        colors[k] = Vector{UnitRange{Int}}(undef, 0)
        coloring_tree = tree_function(centers[k])                
        neighborhoods = inrange(coloring_tree, centers[k], scales[k] * ρ)
        uncolored = Vector{Int}(1 : Nk)
        while colored < Nk
            uncolored = filter(i -> ~nodes_colored[i], uncolored)
            # identify nodes to be colored
            start_index = length(Ps[k]) + 1
            # Nodes that have already been taken 
            # TODO right now we revisit all nodes again when determining which to color next, which is wasteful. 
            nodes_taken = falses(Nk)
            nodes_taken .= nodes_colored
            for center_index in uncolored
                nhd = neighborhoods[center_index]
                if !(nodes_taken[center_index])
                    # Add node to given color
                    push!(Ps[k], center_index)
                    # remove nearby nodes as candidates for the given color.
                    nodes_taken[nhd] .= true
                    nodes_colored[center_index] = true
                end                    
            end
            # After a maximal coloring is obtained, push to colors
            push!(colors[k], start_index : length(Ps[k]))

            # identify sparsity pattern associated to these nodes.
            # Possible savings by ignoring block-upper-triangular part

            # construct tree from centers that were just colored
            sparsity_tree = tree_function(centers[k][Ps[k][colors[k][end]]])

            # for each basis funtion, compute the nearest neighbor among the nodes just colored.
            sparsity_list = nn(sparsity_tree, all_centers)[1]

            sparsity_list = map( x -> (x[2], x[1]), collect(enumerate(sparsity_list))); sort!(sparsity_list)

            # index ranges contains Unit ranges that, when indexing into sparsity_list, pick out the rowvals assigned to the same column 
            index_ranges = Vector{UnitRange{Int}}(undef, length(unique(getindex.(sparsity_list, 1))))
            last_end = 0
            for k = 1 : (length(index_ranges) - 1)
                new_end = findnext(x -> x[1] != sparsity_list[last_end + 1][1], sparsity_list, last_end + 1) - 1
                index_ranges[k] = (last_end + 1) : new_end
                last_end = new_end
            end
            index_ranges[end] = (last_end + 1) : length(sparsity_list)

            # append all row values. Since we ordered sparsity_list lexicographically the values will be matched both to the right column and appear in the right order.
            append!(rowval, getindex.(sparsity_list, 2))
            append!(colptr, colptr[end] .+ cumsum(length.(index_ranges)))
            
            # Update the number of nodes already colored
            colored += length(index_ranges)
        end
        # We now have rowval and colptr that represent the sparsity pattern, with the columns already ordered according to Ps and the rows still in the original ordering.
    end
    # Construct matrix with both rows and columns ordered according to the Ps 
    Psums = vcat(0, cumsum(length.(Ps))[1 : (end - 1)])
    P = mapreduce((x, y) -> x .+ y, vcat, Ps, Psums)
    all_colors = mapreduce((x, y) -> broadcast(z -> z .+ y, x), vcat, colors, Psums)

    𝐋 = tril(SparseMatrixCSC{Float64,Int}(N, N, colptr, rowval, zeros(length(rowval)))[P, :])
    𝐖 = mapreduce((x,y) -> x[:, y], hcat, 𝐖s, Ps)

    𝐌 = Matrix(mapreduce(x -> sum(𝐖[:, x], dims=2), hcat, all_colors))

    all_centers = mapreduce((x,y) -> x[y], vcat, centers, Ps)


    return 𝐋, 𝐖, all_colors, 𝐌, all_centers
end

function plot_domain!(pl, t::Domain)
    Plots.scatter!(pl, getindex.(descendants(t)[1], 1), getindex.(descendants(t)[1], 2))
    Plots.scatter!(pl, [center(t)[1]], [center(t)[2]])
end 

function plot_domain(t::Domain)
    pl = Plots.plot(xlims=(-0.1, 1.1), ylims=(-0.1, 1.1))
    plot_domain!(pl, t)
end

# function that scatters the color-l column of $𝐎$ into the columns of 𝐋 as indicated by l
function scatter!(𝐋, 𝐎, C, l) 
    for index in C[l]
        𝐋.nzval[𝐋.colptr[index] : (𝐋.colptr[index + 1] - 1)] .= 𝐎[𝐋.rowval[𝐋.colptr[index] : (𝐋.colptr[index + 1] - 1)], l]
    end
end

function CSC_leading_column_view(L::SparseMatrixCSC{Tv, Ti}, k) where {Tv, Ti<:Integer}
    return SparseMatrixCSC{Tv, Ti}(L.m, k, unsafe_wrap(Vector{Ti}, pointer(L.colptr), (k + 1)), unsafe_wrap(Vector{Ti}, pointer(L.rowval), (L.colptr[k + 1] - 1)), unsafe_wrap(Vector{Tv}, pointer(L.nzval), (L.colptr[k + 1] - 1)))
end


# TODO: Debug, still doesn't seem to be totally right? 
function reconstruct!(𝐋, 𝐎, 𝐌, C)
    tol = 1e-8
    # Add first color, which doesn't need any peeling
    scatter!(𝐋, 𝐎, C, 1)
    @show length(C)
    scratch_vec = zeros(size(𝐋, 1))
    diag_inv_vec = Vector{Float64}(undef, length(scratch_vec))
    for i in C[1]
        if 𝐋[i, i] > tol
            diag_inv_vec[i] = 1 / 𝐋[i, i]
        else
            # println("Pivot $i close to singular.") 
            diag_inv_vec[i] = 0
        end
    end
    for l = 2 : length(C)
        scratch_view = view(scratch_vec, 1 : (C[l][1] - 1) )
        diag_inv_view = view(diag_inv_vec, 1 : (C[l][1] - 1) )
        # Produce a view covering the matrix reconstructed so far
        # 𝐋_view = view(𝐋, :, 1 : (C[l][1] - 1))
        # 𝐎[:, l] .= 𝐎[:, l] .- 𝐋_view * (diagm(diag(𝐋_view)) \ (𝐋_view' * 𝐌[:, l]))

        𝐋_view = CSC_leading_column_view(𝐋, (C[l][1] - 1))
        mul!(scratch_view, 𝐋_view', view(𝐌, :, l))
        scratch_view .*= diag_inv_view
        mul!(view(𝐎, :, l), 𝐋_view, scratch_view, -1, 1)

        scatter!(𝐋, 𝐎, C, l)

        # add new diagonal entries
        for i in C[l]
            if 𝐋[i, i] > tol
                diag_inv_vec[i] = 1 / 𝐋[i, i]
            else
                # println("Pivot $i close to singular.") 
                diag_inv_vec[i] = 0
            end
        end

    end
    # function pos_sqrt_inv(x)
    #     if x > 0
    #         return 1 / sqrt(x)
    #     else
    #         return zero(x)
    #     end
    # end
    # TODO: Presently seems to allocate dense array?!?!?
    # 𝐋.nzval[𝐋.colptr[k] : (𝐋.colptr[k + 1] - 1)] ./= sqrt(𝐋.nzval[𝐋.colptr[k]])
    for k = 1 : size(𝐋, 2)
        𝐋.nzval[𝐋.colptr[k] : (𝐋.colptr[k + 1] - 1)] .*= sqrt(diag_inv_vec[k])
    end
    # 𝐋 .= 𝐋 ./ sqrt.(diag(𝐋))
end

function plot_column(L, centers, k) 
    pl_out = Plots.plot(xlims=(-0.1, 1.1), ylims=(-0.1, 1.1))
    Plots.scatter!(pl_out, getindex.(centers[k:end], 1), getindex.(centers[k:end], 2), marker_z=vec(log10.(abs.(L[k : end, k]))))
    return pl_out
end