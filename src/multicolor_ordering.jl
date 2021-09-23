using Base: Float64
import SparseArrays.SparseVector
import NearestNeighbors: KDTree, nn, inrange, knn
import LinearAlgebra.Vector
import DataStructures: MutableBinaryMaxHeap, top_with_handle, pop!, update!

function construct_multicolor_ordering(input_array::AbstractVector, ρh::Real, tree_function)
    RT = real_type(eltype(input_array))
    # Vector (colors) of Vectors of supernodes  
    out = Vector{eltype(input_array)}[]

    assigned = falses(length(input_array))
    # While not all nodes are assigned to a color
    tree = tree_function(center.(input_array))
    heap = MutableBinaryMaxHeap(fill(typemax(RT), length(input_array)))
    while !isempty(heap)
        # add new color
        push!(out, typeof(input_array)(undef, 0))
        # reset heap values
        # Presently a little inefficient since we revisit every entry of the multicolor 
        # ordering. Still doesn't change the asymptotic complexity
        for k = 1 : length(input_array)
            # only assigns a value to points that have not already been assigned and thus
            # removed from the heap
            !assigned[k] && (heap[k] = typemax(RT)) 
        end
        # do furthest point sampling, finish if either the heap is empty, or there are no
        # sufficiently distant points left 
        while !isempty(heap) && first(heap) > 2 * ρh
            # get the id of the new pivot
            ~, top_id = top_with_handle(heap)
            # remove the new pivot from the heap
            pop!(heap); assigned[top_id] = true; push!(out[end], input_array[top_id])
            # add the new pivot to the 
            # obtain the affected nodes and their distance to the pivot 
            number_affected_nodes = length(inrange(tree, center(input_array[top_id]), 2 * ρh))
            affected_nodes, distances = knn(tree, center(input_array[top_id]), number_affected_nodes)
            # Update the distances of nodes that could be affedcted. 
            for (node, dist) in zip(affected_nodes, distances)
                # update distance if node is still in heap
                !assigned[node] && update!(heap, node, dist)
            end
        end
    end
    return out
end

# function that can directly take an array of arrays (corresponding to different scales) as supernodes
function construct_multicolor_ordering(input_arrays::AbstractVector{<:AbstractVector}, ρh::AbstractVector{<:Real}, tree_function)
    # Both lengths should be equal to total number of scales
    @assert length(ρh) == length(input_arrays)
    q = length(ρh)
    out = Vector{typeof(input_arrays)}(undef, q)
    for k = 1 : q
        out[k] = construct_multicolor_ordering(input_arrays[k], ρh[k], tree_function)
    end
    return vcat(out...)
end