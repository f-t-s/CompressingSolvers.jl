using SparseArrays: getcolptr, getrowval, getnzval
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

# ordering is a multicolor ordering of basis functions
function reconstruct(ordering, row_centers, measurement, tree_function=KDTree)
    I, J = sparsity_set(ordering, row_centers, tree_function)
    L = sparse(I, J, zeros(eltype(measurement), length(I)))
    # going through the columns of the measurements  
    temp_col = zeros(L, 2)
    temp_row = zeros(L, 1)
    for k = 1 : size(measurement, 2)
        mul!(view(temp, 1 : (k - 1)), view(L, :, 1 : k - 1))
        mul!(temp_col, view(L, :, 1 : k - 1)') 
    end
end