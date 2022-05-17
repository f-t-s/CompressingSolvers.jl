using SparseArrays: getcolptr, getrowval, getnzval, sparse
using LinearAlgebra: dot, Factorization, mul!
using Distances: Euclidean 
import Base.* 

# a simple struct that wraps the reconstruction 
struct Reconstruction
    L::SparseMatrixCSC
end

function *(rc::Reconstruction, v)
    return rc.L * (rc.L' * v)
end

# A function that takes in a reconstruction problem and returns the Reconstruction
function reconstruct(pb::ReconstructionProblem, ρ, h=0.5)
    if typeof(pb.distance) == Euclidean
        tree_function(x) = KDTree(x, pb.distance)
    else
        tree_function(x) = BallTree(x, pb.distance)
    end
    # create a domain hierarchy 
    domain_hierarchy = gather_hierarchy(create_hierarchy(pb.domains, h, tree_function))
    # compute approximations of the scales on each level
    scales = [maximum(approximate_scale(center.(domain_hierarchy[k]), tree_function)) for k = 1 : length(domain_hierarchy)]
    # compute Haar-like bais functions from domain hierarchy 
    basis_functions = compute_basis_functions(first(domain_hierarchy)) 
    # computing the multicolor ordering
    multicolor_ordering = construct_multicolor_ordering(basis_functions, scales, tree_function)
    # Forms the measurement matrix
    𝐌 = form_measurement_matrix(multicolor_ordering)
    # Performs the measurement
    𝐎 = pb.ω(𝐌)
    return Reconstruction(reconstruct(multicolor_ordering, center.(pb.domains), 𝐌, 𝐎, tree_function))
end 

# function that takes in a multicolor colors and returns the corresponding sparsity sets
# each entry of the variable "colors" contains a different color,represented as iterable 
# collection of basis functions 
# The input variable "row_bases" contains an array of basis functions that are assigned to the 
# different basis elements in "column_bases", according to the coloring of column_bases 
# given by colors
function sparsity_set(colors,  
                      row_centers::AbstractVector,
                      tree_function)
    I = Int[]
    J = Int[]
    offset = 0
    for color in colors
        column_centers = center.(color)
        tree = tree_function(reduce(hcat, column_centers))
        neighbors = nn(tree, row_centers)[1]
        # the i-th entry of neighbors contains the column j + offset that the ith row 
        # is assigned to
        for (i, j) in enumerate(neighbors)
            push!(I, i)
            push!(J, j + offset)
        end
        offset += length(color)
    end
    return I, J
end

# function that takes in a column and scatters its values into the columns given by 
# color_range sparse matrix L
function scatter_color!(L, vals, color_range)
    rowval = getrowval(L)
    colptr = getcolptr(L)
    nzval = getnzval(L)
    for j in first(color_range) : last(color_range)
        for i_index in colptr[j] : (colptr[j + 1] - 1)
            i = rowval[i_index]
            nzval[i_index] = vals[i]
        end
    end
end 

function form_measurement_matrix(ordering) 
    return reduce(hcat, [Vector(sum(coefficients.(color))) for color in ordering])
end

function measure(A::Factorization, measurement_matrix)
    return A \ measurement_matrix
end

function update_active_L(active_nzval, active_colptr, active_rowval, L, color_range)
    added_indices_range = L.colptr[first(color_range)] : (L.colptr[last(color_range) + 1] - 1)
    # appending the next elements to the active L
    append!(active_nzval, view(L.nzval, added_indices_range))
    append!(active_rowval, view(L.rowval, added_indices_range))
    append!(active_colptr, view(L.colptr, (first(color_range) + 1) : (last(color_range) + 1)))
    return SparseMatrixCSC(size(L, 1), last(color_range), active_colptr, active_rowval, active_nzval)
end

# ordering is a multicolor ordering of basis functions
function reconstruct(ordering, row_centers, measurement_matrix, measurement_results, tree_function)
    @assert length(ordering) == size(measurement_results, 2)
    I, J = sparsity_set(ordering, row_centers, tree_function)
    L = sparse(I, J, zeros(eltype(measurement_results), length(I)))
    # L_empty = L[:, 1:0]
    active_nzval, active_colptr, active_rowval = [f(L[:, 1 : 0]) for f in [getnzval, getcolptr, getrowval]]
    # going through the columns of the measurements  
    temp = zeros(size(L, 2))
    offset = 0 
    for k = 1 : size(measurement_results, 2)
        # Subtract the existing factor from measurement
        # TODO: avoid temporary allocations
        active_L = SparseMatrixCSC(size(L, 1), offset, active_colptr, active_rowval, active_nzval)

        @views mul!(temp[1 : offset], active_L', measurement_matrix[:, k])
        @views mul!(measurement_results[:, k], active_L, temp[1 : offset], -1, 1)
        # mul!(view(temp, 1 : (k - 1)), view(L, :, 1 : k - 1))
        # mul!(temp_col, view(L, :, 1 : k - 1)') 

        # the range of column indices corresponding to the present color
        color_range = ((1 + offset) : (length(ordering[k]) + offset))
        active_L = update_active_L(active_nzval, active_colptr, active_rowval, L, color_range)
        # assign the values of the treated measurements to the corresponding columns of L
        scatter_color!(active_L, view(measurement_results, :, k), color_range)

        # normalize the column

        for (k_column, basis_function) in zip(color_range, ordering[k])
            @views active_L[:, k_column] ./= sqrt(dot(coefficients(basis_function), active_L[:, k_column]))
        end

        # update the offset that allows to assign column indices to entries of a given color
        offset += length(ordering[k])
    end

    return SparseMatrixCSC(size(L, 1), offset, active_colptr, active_rowval, active_nzval)
end