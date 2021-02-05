import SparseArrays.SparseVector
import NearestNeighbors: KDTree, nn, inrange
import LinearAlgebra.Vector

# Abstract type for super nodes
abstract type AbstractSuperNode{PT} end
# A struct that contains a supernode, the corresponding basis vectors, and their identity 
struct SuperNodeBasis{PT} <: AbstractSuperNode{PT}
    center::PT
    basis_vectors::Vector{SparseVector{eltype(PT), Int}}
end

# A struct that contains a supernode for the DOFs
struct SuperNodeDomain{PT} <: AbstractSuperNode{PT}
    center::PT
    domains::Vector{Domain{PT}}
end

function SuperNodeBasis(center::PT, basis_vectors::Vector{SparseVector{RT, Int}}) where {RT<:Real,PT<:AbstractArray{RT}}
    return SuperNodeBasis{PT}(center, basis_vectors)
end

function SuperNodeDomain(center::PT, domains::Vector{Domain{PT}}) where {PT<:AbstractArray{<:Real}}
    return SuperNodeDomain{PT}(center, domains)
end


function get_center(in::AbstractSuperNode)
    return in.center
end

# Construct supernodes about aggregation centers provided by the user
# centers is the vector of centers of basis functions, and 
function construct_supernodes(aggregation_centers, basis_vectors, centers, tree_function=KDTree)
    # allocating output array
    out = Vector{SuperNodeBasis{eltype(centers)}}(undef, length(aggregation_centers))
    # Constructing themembership lists of the different supernodes, by assigning them to the closest aggregation center
    member_lists = construct_member_lists(nn(tree_function(aggregation_centers), centers)[1])
    # Creating the new SuperNode
    for (k, list) in enumerate(member_lists)
        out[k] = SuperNodeBasis(aggregation_centers[k], basis_vectors[list])
    end
    return out 
end

# Construct supernodes about aggregation centers provided by the user
# centers is the vector of centers of basis functions, and 
function construct_supernodes(aggregation_centers, domains::AbstractVector{<:Domain}, tree_function=KDTree)
    centers = center.(domains)
    # allocating output array
    out = Vector{SuperNodeDomain{eltype(centers)}}(undef, length(aggregation_centers))
    # Constructing themembership lists of the different supernodes, by assigning them to the closest aggregation center
    member_lists = construct_member_lists(nn(tree_function(aggregation_centers), centers)[1])
    # Creating the new SuperNode
    for (k, list) in enumerate(member_lists)
        out[k] = SuperNodeDomain(aggregation_centers[k], domains[list])
    end
    return out 
end


function construct_multicolor_ordering(input_supernodes::AbstractArray{<:SuperNodeBasis}, ρh, tree_function=KDTree)
    # Vector (colors) of Vectors of supernodes  
    out = Vector{typeof(input_supernodes)}(undef, 0)
    assigned = falses(length(input_supernodes))
    # While not all nodes are assigned to a color
    tree = tree_function(get_center.(input_supernodes))
    while sum(length.(out)) < length(input_supernodes)
        ruled_out = falses(size(assigned))
        # add new color
        push!(out, typeof(input_supernodes)(undef, 0))
        for (k, node) in enumerate(input_supernodes)
            #Check that node was not ruled out or assigned yet
            if !(ruled_out[k] || assigned[k])
                # Add new node to present color
                push!(out[end], node)
                # Make note that node was assigned to color
                assigned[k] = true
                # Make note that its neighbors are not allowed to be assigned to the same color
                ruled_out[inrange(tree, get_center(node), ρh)] .= true
            end
        end
    end
    return out
end

# function that can directly take an array of arrays (corresponding to different scales) as supernodes
function construct_multicolor_ordering(input_supernodes::AbstractArray{<:AbstractArray{<:SuperNodeBasis}}, ρh::AbstractArray{<:Real}, tree_function=KDTree)
    # Both lengths should be equal to total number of scales
    @assert length(ρh) == length(input_supernodes)
    q = length(ρh)
    out = Vector{typeof(input_supernodes)}(undef, q)
    for k = 1 : q
        out[k] = construct_multicolor_ordering(input_supernodes[k], ρh[k], tree_function)
    end
    return out
end 