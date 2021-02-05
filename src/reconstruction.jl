import LinearAlgebra: Matrix, Cholesky, cholesky, mul!, ldiv!
import Base.size
# Defining an alias for matrices that are realized as resized contiguous view into a buffer
const ContiguousBufferedMatrix{T} = Base.ReshapedArray{T,2,SubArray{Float64,1,Array{T,1},Tuple{UnitRange{Int64}},true},Tuple{}}

# A supernodal column that contains 
struct SupernodalColumn{RT}
    data::Vector{ContiguousBufferedMatrix{RT}}
    buffer::Vector{RT}
    row_supernodes::Vector{Vector{Int}}
end

# Constructs a supernodal column from a dense matrix
function SupernodalColumn(𝐌, row_supernodes)
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
    return SupernodalColumn{eltype(𝐌)}(data, buffer, row_supernodes)
end

function Matrix(col::SupernodalColumn)
    M = sum(length.(col.row_supernodes))
    N = Int(length(col.buffer) / M)
    out = zeros(eltype(col.buffer), M, N)
    # Iterate over all block matrices in data 
    for (k, block) in enumerate(col.data)
        # Iterate over all indices in the block matrix
        for c in CartesianIndices(block)
            # Transform the i-index from the local index c[1] to the global index, as read of from the corresponding row_supernode
            out[col.row_supernodes[k][c[1]], c[2]] = block[c]
        end
    end
    return out
end

# We store the factorization as an LDL factorization in column-major ordering
struct SupernodalFactorization{RT}
    # This is the underlying buffer storing all nonzeros of the factor
    # The nonzeros appear column-by-column and, within each column, supernode by supernode

    # The (supernodal) sparse matrix strcture, analogu to SparseMatrixCSC
    rowval::Vector{Int}
    colptr::Vector{Int} 
    nnz::Vector{ContiguousBufferedMatrix{RT}}
    # For now, we don't buffer the diagonal array
    diag::Vector{Cholesky{RT,Matrix{RT}}}

    # The member lists of the underlying supernodes 
    row_supernodes::Vector{Vector{Int}}
    column_supernodes::Vector{Vector{Int}}
    
    # The buffer underlying nnz
    buffer::Vector{RT}
end

# Constructor for the supernodal factorization
function SupernodalFactorization{RT}(I::AbstractArray{Int}, J::AbstractArray{Int}, row_supernodes, column_supernodes) where {RT<:Real}
    @assert length(I) == length(J)
    M = sum(length.(row_supernodes))
    N = sum(length.(column_supernodes))
    @assert M == N
    # Check that row and column supernode indices are valid 
    @assert sort(vcat(row_supernodes...)) == 1 : M
    @assert sort(vcat(column_supernodes...)) == 1 : N
    m = length(row_supernodes)
    n = length(column_supernodes)
    # using the existing sparse functionality to construct rowval and colptr 
    temp_L = sparse(I, J, falses(length(I)), m, n) 
    rowval = temp_L.rowval
    colptr = temp_L.rowval
    # # New I and J don't have duplicates any more
    I, J, ~ = findnz(temp_L)
    # Length of buffer is equal to the sum of the product of the length of the supernodes involved in each entry
    row_lengths = length.(row_supernodes) 
    column_lengths = length.(column_supernodes) 
    cum_sum_product_lengths = cumsum(row_lengths .* column_lengths)
    buffer = Vector{RT}(undef, cum_sum_product_lengths[end])
    cum_sum_product_lengths = 1 .+ vcat([0], cum_sum_product_lengths)
    nnz = Vector{ContiguousBufferedMatrix{RT}}(undef, length(rowval))
    for k = 1 : (length(cum_sum_product_lengths) - 1)
        nnz[k] = reshape(view(buffer, cum_sum_product_lengths[k] : (cum_sum_product_lengths[k + 1]) - 1), row_lengths[k], column_lengths)
    end
    diag = Vector{Cholesky{RT,Matrix{RT}}}(undef, n)
    return SupernodalFactorization{RT}(rowval, colptr, nnz, diag, row_supernodes, column_supernodes, buffer) 
end

# Multiply the matrix implied by a supernodal factorization with a supernodal column,
function partial_multiply!(out::SupernodalColumn, 𝐋::SupernodalFactorization, in; max_k=length(𝐋.column_supernodes), scratch=SupernodalColumn(zeros(eltype(𝐋.buffer), size(sum(length.(𝐋.column_supernodes[1 : max_k])), size(in.data[1], 2))), 𝐋.column_supernodes[1 : max_k]))

    scratch.buffer .= 0.0
    # computing scratch = 𝐋' * in
    for k = 1 : max_k
        for i_index = 𝐋.colptr[k] : (𝐋.colptr[k + 1] - 1)
            i = 𝐋.rowval[i_index]
            mul!(scratch.data[k], 𝐋.nnz[i_index]', in.data[i], 1, 1)
        end
    end

    # Dividing by the diagonal 
    for k = 1 : max_k
        ldiv!(𝐋.diag[k], scratch[k])
    end

    out.buffer .= 0.0
    # Computing out = L * scratch
    for k = 1 : max_k
        for i_index = 𝐋.colptr[k] : (𝐋.colptr[k + 1] - 1)
            i = 𝐋.rowval[i_index]
            mul!(out[i], 𝐋.nnz[i_index], scratch.data[k], 1, 1)
        end
    end
end

# scatters the Supernodal Column 𝐌 into the columns of 𝐋 given by color
function scatter!(𝐋::SupernodalFactorization, 𝐌::SupernodalColumn ,color::AbstractVector{Int}, ) 
    # We check that each children supernode appears at most once in the sparsity pattern of the columns of the color
    @assert (𝐋.rowval[𝐋.colptr[color[1]]: 𝐋.colptr[color[end]]]) == unique(𝐋.rowval[𝐋.colptr[color[1]]: 𝐋.colptr[color[end]]])
    for index = 𝐋.colptr[color[1]] : 𝐋.colptr[color[end]]
        # Setting the nnz values
        𝐋.nnz[index] .= view(𝐌.data[𝐋.rowval[index]], :, 1 : size(𝐋.nnz[index], 2))
    end
end