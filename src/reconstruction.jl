using SparseArrays: getcolptr, getrowval, getnzval, sparse
using LinearAlgebra: dot, Factorization, mul!
# function that takes in a multicolor colors and returns the corresponding sparsity sets
# each entry of the variable "colors" contains a different color,represented as iterable 
# collection of basis functions 
# The input variable "row_bases" contains an array of basis functions that are assigned to the 
# different basis elements in "column_bases", according to the coloring of column_bases 
# given by colors
function sparsity_set(colors,  
                      row_centers::AbstractVector,
                      tree_function=KDTree)
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

# ordering is a multicolor ordering of basis functions
function reconstruct(ordering, row_centers, measurement_matrix, measurement_results, tree_function=KDTree)
    @assert length(ordering) == size(measurement_results, 2)
    I, J = sparsity_set(ordering, row_centers, tree_function)
    L = sparse(I, J, zeros(eltype(measurement_results), length(I)))
    # going through the columns of the measurements  
    temp = zeros(size(L, 2))
    offset = 0 
    for k = 1 : size(measurement_results, 2)
        # Subtract the existing factor from measurement
        # TODO: avoid temporary allocations
        temp_L = L[:, 1 : offset]
        @views mul!(temp[1 : offset], temp_L', measurement_matrix[:, k])
        @views mul!(measurement_results[:, k], temp_L, temp[1 : offset], -1, 1)
        # mul!(view(temp, 1 : (k - 1)), view(L, :, 1 : k - 1))
        # mul!(temp_col, view(L, :, 1 : k - 1)') 

        # the range of column indices corresponding to the present color
        color_range = ((1 + offset) : (length(ordering[k]) + offset))
        # assign the values of the treated measurements to the corresponding columns of L
        scatter_color!(L, view(measurement_results, :, k), color_range)

        # normalize the column
        for (k_column, basis_function) in zip(color_range, ordering[k])
            @views L[:, k_column] ./= sqrt(dot(coefficients(basis_function), L[:, k_column]))
        end

        # update the offset that allows to assign column indices to entries of a given color
        offset += length(ordering[k])
    end
    return L
end