import LinearAlgebra: Matrix, Cholesky, cholesky, mul!, ldiv!
import Base: size, getindex, enumerate, iterate
import SparseArrays: SparseVector, SparseMatrixCSC, nonzeros, rowvals, getcolptr, sparse, sparsevec, nnz
# Defining an alias for matrices that are realized as resized contiguous view into a buffer
const ContiguousBufferedMatrix{T} = Base.ReshapedArray{T,2,SubArray{Float64,1,Array{T,1},Tuple{UnitRange{Int64}},true},Tuple{}}

# Empty constructor used for initializing arrays
function ContiguousBufferedMatrix{T}() where T
    return reshape(view(zeros(T, 0), 1:0), 0, 0)
end

abstract type  AbstractSupernodalArray{Tv, N} end 
abstract type  AbstractSupernodalSparseArray{Tv, N} <: AbstractSupernodalArray{Tv,N} end 

function data(in::AbstractSupernodalArray)
    return in.data
end

function nonzeros(in::AbstractSupernodalSparseArray)
    return nonzeros(data(in))
end

function rowvals(in::AbstractSupernodalSparseArray)
    return rowvals(data(in))
end

function getcolptr(in::AbstractSupernodalSparseArray)
    return getcolptr(data(in))
end

function getindex(in::AbstractSupernodalSparseArray, I...)
    return getindex(data(in), I...)
end

function enumerate(in::AbstractSupernodalArray)
    enumerate(data(in))
end

# A supernodal column that contains  
struct SupernodalVector{RT} <: AbstractSupernodalArray{RT,1}
    data::Vector{ContiguousBufferedMatrix{RT}}
    buffer::Vector{RT}
    row_supernodes::Vector{Vector{Int}}
end

# Constructs a supernodal column from a dense matrix and supernodal ids
function SupernodalVector(𝐌, row_supernodes::AbstractVector{<:AbstractVector{Int}})
    buffer = zeros(eltype(𝐌), length(𝐌))
    data = Vector{ContiguousBufferedMatrix{eltype(𝐌)}}(undef, length(row_supernodes))
    N = size(𝐌, 2)
    start_index = 1 
    for (k, node) in enumerate(row_supernodes)
        data[k] = reshape(view(buffer, start_index : (start_index + length(node) * N - 1)), length(node), N)
        start_index += length(node) * N
    end
    for (k, block) in enumerate(data)
        # Iterate over all indices in the block matrix
        for c in CartesianIndices(block)
            # Transform the i-index from the local index c[1] to the global index, as read of from the corresponding row_supernode
            block[c] = 𝐌[row_supernodes[k][c[1]], c[2]]
        end
    end
    return SupernodalVector{eltype(𝐌)}(data, buffer, row_supernodes)
end

# A sparse supernodal column, which will be used for storing the basis functions in supernodal form
struct SupernodalSparseVector{RT} <: AbstractSupernodalSparseArray{RT,1}
    data::SparseVector{ContiguousBufferedMatrix{RT}, Int}
    buffer::Vector{RT}
    row_supernodes::Vector{Vector{Int}}
end

function size(in::SupernodalSparseVector)
    return (sum(length.(in.row_supernodes)), size(first(in.data.nzval), 2))
end

function size(in::SupernodalSparseVector, dim)
    return (sum(length.(in.row_supernodes)), size(first(in.data.nzval), 2))[dim]
end

# Construct a supernodal sparse vector from a sparse matrix 
function SupernodalSparseVector(mat::SparseMatrixCSC, row_supernodes::Vector{Vector{Int}})
    M, N = size(mat)
    @assert sum(length.(row_supernodes)) == M
    @assert sort(vcat(row_supernodes...)) == 1 : M
    # A vector such that row_supernodes[supernode_dict[k][1]][supernode_dict[k][2]] = k
    supernode_dict = Vector{Tuple{Int,Int}}(undef, M)
    for (i, node) in enumerate(row_supernodes)
        for (j, k) in enumerate(node)
            supernode_dict[k] = (i, j)
        end
    end
    I_el, J_el, S_el = findnz(mat)
    # Re-expressing the dofs in terms of the supernodes
    I_super = supernode_dict[I_el]
    # keeping track of those supernodes that appear in the sparsity pattern
    # We preallocate the array that will hold the matrix elements of the sparse vector
    S_super = Vector{ContiguousBufferedMatrix{eltype(mat)}}(undef, length(unique(getindex.(I_super, 1))))
    # We sum up the lengths of the supernodes in the sparsity pattern and multiply them with the number of entries per row.
    buffer = Vector{eltype(mat)}(undef, sum(length.(row_supernodes[unique(getindex.(I_super, 1))])) * N)
    # we are sorting the entries according by the supernode that they form part of
    sp = sortperm(I_super)
    # reordering the sparsity entries
    I_super = I_super[sp]
    I_el = I_el[sp]
    J_el = J_el[sp]
    S_el = S_el[sp]

    # We now extract those supernodes that appear in the sparsity pattern and recreate the dict, with the first index indicating the position of the supernode among the used supernodes
    used_supernode_indices = unique(getindex.(I_super, 1))
    lookup = Dict(used_supernode_indices .=> 1 : length(used_supernode_indices))
    used_row_supernodes = row_supernodes[used_supernode_indices]

    # Setting up the buffer and nonzeros
    offset = 0
    for (k, node) in enumerate(used_row_supernodes)
        S_super[k] = reshape(view(buffer, offset .+ (1 : (length(node)* N))), length(node), N)
        # filling the new array with zeros
        S_super[k] .= 0
        # updating the offset 
        offset += length(node) * N 
    end

    for (i_super, j, s) in zip(I_super, J_el, S_el)
        # in the supernode \# i_super[1], we add s to the entry (i_super[2], j)
        S_super[lookup[i_super[1]]][i_super[2], j] += s
    end
    return SupernodalSparseVector{eltype(mat)}(sparsevec(used_supernode_indices, S_super), buffer, row_supernodes)
end

function SupernodalSparseVector(node::SuperNodeBasis, row_supernodes::AbstractVector{<:AbstractVector{Int}})
    return SupernodalSparseVector(hcat(coefficients.(basis_functions(node))...), row_supernodes)
end

# inverse of the constructor of the sparsesupernodalvector
function SparseMatrixCSC(in::SupernodalSparseVector)
    out_I = Vector{Int}(undef, 0)
    out_J = Vector{Int}(undef, 0)
    out_S = Vector{eltype(in.buffer)}(undef, 0)
    for (i, mat) in zip(findnz(data(in))...)
        node = in.row_supernodes[i]
        for (i, j) in Tuple.(CartesianIndices(mat))
            push!(out_I, node[i])
            push!(out_J, j)
            push!(out_S, mat[i, j])
        end
    end
    return sparse(out_I, out_J, out_S, sum(length.(in.row_supernodes)), size(first(data(in).nzval), 2))
end

# Constructs a supernodal column from a dense matrix and a list of domain supernodes
function SupernodalVector(𝐌, row_supernodes::AbstractVector{<:SuperNodeDomain})
    return SupernodalVector(𝐌, id.(domains(row_supernodes)))
end

function Matrix(col::SupernodalVector)
    M = sum(length.(col.row_supernodes))
    N = Int(length(col.buffer) / M)
    out = zeros(eltype(col.buffer), M, N)
    # Iterate over all block matrices in data 
    for (k, block) in enumerate(col)
        # Iterate over all indices in the block matrix
        for c in CartesianIndices(block)
            # Transform the i-index from the local index c[1] to the global index, as read of from the corresponding row_supernode
            out[col.row_supernodes[k][c[1]], c[2]] = block[c]
        end
    end
    return out
end

# We store the factorization as an LL' factorization in column-major ordering
struct SupernodalFactorization{RT}<:AbstractSupernodalSparseArray{RT,2}
    # This is the underlying buffer storing all nonzeros of the factor
    # The nonzeros appear column-by-column and, within each column, supernode by supernode

    # The (supernodal) sparse matrix structure
    data::SparseMatrixCSC{ContiguousBufferedMatrix{RT},Int}

    # The member lists of the underlying supernodes 
    row_supernodes::Vector{Vector{Int}}

    # column_supernodes, stored as a vector of vectors of sparse supernodal vectors representing the hcated basis functions in terms of the row_supernodes
    # The outermost vector loops over the different colos of the multicolor ordering.
    column_supernodes::Vector{Vector{SupernodalSparseVector{RT}}}
    
    # The buffer underlying nnz
    buffer::Vector{RT}
end

# Constructor for the supernodal factorization
function SupernodalFactorization{RT}(I::AbstractVector{Int}, J::AbstractVector{Int}, row_supernodes::AbstractVector{<:AbstractVector{Int}}, column_supernodes::AbstractVector{<:AbstractVector{SupernodalSparseVector{RT}}}) where {RT<:Real}
    lengths_column_supernodes = size.(vcat(vcat(column_supernodes...)...), 2)
    @assert length(I) == length(J)
    M = sum(length.(row_supernodes))
    N = sum(lengths_column_supernodes)


    @assert M == N
    # Check that row and column supernode indices are valid 
    @assert sort(vcat(row_supernodes...)) == 1 : M
    m = length(row_supernodes)
    n = length(lengths_column_supernodes)
    # using the existing sparse functionality to construct rowval and colptr 
    data = sparse(I, J, fill(ContiguousBufferedMatrix{RT}(), length(I)))
    # Length of buffer is equal to the sum of the product of the length of the supernodes involved in each entry
    # After removind duplicates 
    I, J, ~ = findnz(data)
    row_lengths = length.(row_supernodes) 
    cum_sum_product_lengths = cumsum(row_lengths[I] .* lengths_column_supernodes[J])
    buffer = Vector{RT}(undef, cum_sum_product_lengths[end])
    cum_sum_product_lengths = 1 .+ vcat([0], cum_sum_product_lengths)
    for k = 1 : (length(cum_sum_product_lengths) - 1)
        data.nzval[k] = reshape(view(buffer, cum_sum_product_lengths[k] : (cum_sum_product_lengths[k + 1]) - 1), row_lengths[I[k]], lengths_column_supernodes[J[k]])
    end
    return SupernodalFactorization{RT}(data, row_supernodes, column_supernodes, buffer) 
end

# Constructs a supernodal factorization from a multicolor ordering 
function SupernodalFactorization(multicolor_ordering::AbstractVector{<:AbstractVector{SuperNodeBasis{PT,RT}}}, domain_supernodes::AbstractVector{<:SuperNodeDomain}, tree_function=KDTree) where {PT,RT<:Real}
    row_supernodes = [id.(domains(domain_supernodes[k])) for k = 1 : length(domain_supernodes)]
    column_supernodes = Vector{Vector{SupernodalSparseVector{RT}}}(undef, length(multicolor_ordering))
    for k = 1 : length(column_supernodes)
        column_supernodes[k] = Vector{SupernodalSparseVector{RT}}(undef, length(multicolor_ordering[k]))
        for (l, node) in enumerate(multicolor_ordering[k])
            column_supernodes[k][l] = SupernodalSparseVector(node, row_supernodes)
        end
    end
    # Gather the lengths of the domain supernodes 
    # obtain the (indices of) the row_supernodes
    I = Vector{Int}(undef, 0)
    J = Vector{Int}(undef, 0)
    # the offsets used to transform position within a given color to the position in the full sparse array
    offsets = [0; cumsum(length.(multicolor_ordering))[ 1 : (end - 1)]]
    # iterate over all colors 
    for (offset, color) in zip(offsets, multicolor_ordering)
        # for each domain supernode, compute its nearest neighbor within the present color-- the basis_supernode that it will be assigned to.
        neighbors = nn(tree_function(center.(color)), center.(domain_supernodes))[1]
        # Each entry of neighbors contains the id (with respect to the array color) of the column that it is added to
        for (i, j_color) in enumerate(neighbors)
            push!(I, i) 
            push!(J, offset + j_color) 
        end
    end
    SupernodalFactorization{RT}(I, J, row_supernodes, column_supernodes)
end

function supernodal_size(𝐋::SupernodalFactorization)
    return (length(𝐋.row_supernodes), length(𝐋.diag))
end

function supernodal_size(𝐋::SupernodalFactorization, dim)
    return supernodal_size(𝐋)[dim]
end

function inner_lengths(𝐋::SupernodalFactorization)
    return length.(𝐋.diag)
end


# Multiply the matrix implied by a supernodal factorization with a supernodal vector,
function partial_multiply!(out::SupernodalVector, 𝐋::SupernodalFactorization, in::SupernodalVector; max_k=supernodal_size(𝐋, 2), scratch=SupernodalVector(zeros(eltype(𝐋.buffer), sum(inner_lengths(𝐋)[1 : max_k]), size(data(in)[1], 2)), inner_lengths(𝐋)[1 : max_k]))

    scratch.buffer .= 0.0
    # computing scratch = 𝐋' * in
    for k = 1 : max_k
        for i_index = getcolptr(𝐋)[k] : (getcolptr(𝐋)[k + 1] - 1)
            i = rowvals(𝐋)[i_index]
            mul!(scratch[k], nonzeros(𝐋)[i_index]', in[i], 1, 1)
        end
    end

    out.buffer .= 0.0
    # Computing out = L * scratch
    for k = 1 : max_k
        for i_index = getcolptr(𝐋)[k] : (getcolptr(𝐋)[k + 1] - 1)
            i = rowvals(𝐋)[i_index]
            mul!(out[i], nonzeros(𝐋)[i_index], scratch[k], 1, 1)
        end
    end
end

# scatters the Supernodal Column 𝐌 into the columns of 𝐋 given by color
function scatter!(𝐋::SupernodalFactorization, 𝐌::SupernodalVector ,color::AbstractVector{Int}, ) 
    # We check that each children supernode appears at most once in the sparsity pattern of the columns of the color
    @assert (rowvals(𝐋)[getcolptr(𝐋)[color[1]]: getcolptr(𝐋)[color[end]]]) == unique(rowvals(𝐋)[getcolptr(𝐋)[color[1]]: getcolptr(𝐋)[color[end]]])
    for index = getcolptr(𝐋)[color[1]] : (getcolptr(𝐋)[color[end] + 1] - 1)
        # Setting the nzval
        nonzeros(𝐋)[index] .= view(𝐌[rowvals(𝐋)[index]], :, 1 : size(nonzeros(𝐋)[index], 2))
    end
    # Setting the diagonal values 
    # TODO: Add memory of supernodes to SupernodalFactorization
    matrix_𝐌 = Matrix(𝐌)
end

# Finish function that creates the measurement matrix
function create_measurement_matrix(multicolor_ordering::AbstractVector{<:AbstractVector{SuperNodeBasis{PT,RT}}}) where {PT<:AbstractArray{<:Real}, RT<:Real}
    for color in multicolor_ordering

    end
end
