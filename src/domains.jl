import NearestNeighbors: KDTree, inrange, nn
import StaticArrays.SVector
import LinearAlgebra.norm
import Plots: scatter!, plot

# This file contains the definitions for nested partitions 

# an element of a partition with coordinate center represented by PT
struct Domain{PT}
    # Centroid of partition. 
    center::PT 

    # The "weight" of the domain, which is use to compute orthogonalization of the basis functions and the centroids.
    weight::eltype(PT)

    # The list of the children on the next finer scale.
    children::Vector{Domain{PT}}

    # an id to keep track of the partition elements
    id::Int
end


function weight(t::Domain)
    return t.weight
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

# Returns a list of all the subdomains of a given domain, recursively
function subdomains(t::Domain{PT}) where {PT<:AbstractVector}
    if iselementary(t::Domain)
        return [t]
    else 
        out = Vector{Domain{PT}}(undef, 0)
        for ch in children(t)
            append!(out, subdomains[out])
        end
        return out
    end
end

# construct a new domain from a list of children
function Domain(input_children::AbstractVector{Domain{PT}}, id) where {PT<:AbstractVector}
    out_center = zero(PT)
    out_weight = 0
    out_children = copy(input_children)
    for t in input_children
        out_weight += weight(t)
        out_center += center(t) * weight(t)
    end
    out_center /= out_weight
    return Domain{PT}(out_center, out_weight, out_children, id)
end

# returns a copy of the input domain that has the additional children
# in the array child
function add_children(domain::DT, new_children::AbstractVector{DT}) where DT<:Domain
    return Domain(vcat(children(domain), new_children), id(domain))
end

# construct an elementary domain from a coordinate
function Domain(input_coordinates::PT, id, weight=1) where {PT<:AbstractVector}
    return Domain{PT}(input_coordinates, weight, Vector{Domain{PT}}(undef, 0), id)
end

# construct a list of elementary domains from the columns of a matrix
# can use dims keyword to instead construct them from rows of matrix.
function array2domains(in::AbstractMatrix{<:Real}, weights=1; dims=1) 
    if dims == 2    
        in = transpose(in)
    elseif dims != 1 
        error("Invalid keyword argument for dims")
    end
    @assert typeof(weights) <: Real || length(weights) == size(in, 2)
    weights = weights .* ones(size(in, 2))
    d, N = size(in)
    out = Vector{Domain{SVector{d, eltype(in)}}}(undef, N)
    for k = 1 : N 
        out[k] = Domain(SVector{d}(in[:, k]), k)
    end
    return out
end


# We define a domain as elementary if it does not have any children. Note that this is slightly different than demanding it to consist of a single coordinate domain
function iselementary(t::Domain)
    return isempty(t.children)
end

function approximate_diameter(centers::AbstractVector{<:AbstractVector}) 
    mn = sum(centers) / length(centers)
    return maximum(norm.(repeat([mn], length(centers)) - centers))
end

# Takes in a vector of domains and returns a list of all the elementary domains that are among its descendants
function gather_descendants(domains::AbstractVector{<:Domain})
    out = Vector{eltype(domains)}(undef, 0)
    for domain in domains
        if iselementary(domain) 
            push!(out, domain)
        else
            append!(out, gather_descendants(children(domain)))
        end
    end
    return out
end


# Helper functions for clustering
# The Vector memberships contains elements of ids
# memberships[i] = j signifies that the i-th member is associated 
# to j
function construct_member_lists(memberships_abstract)
    # Catching the special case where there are not membership relations
    if isempty(memberships_abstract)
        return Int[]
    else 
        member_lists = Vector{Tuple{Int, Vector{Int}}}(undef, maximum(memberships_abstract))
        # Initialize with empty array
        for k = 1 : length(member_lists)
            member_lists[k] = (k, Vector{Int}(undef, 0))
        end
        for k = 1 : length(memberships_abstract)
            push!(member_lists[memberships_abstract[k]][2], k)
        end
        # removes all entries corresponding to empty member lists
        member_lists = member_lists[findall(.!isempty.(getindex.(member_lists, 2)))]

        return member_lists
    end
end

# A function to cluster a list of points around centers chosen from among them, that are at least scale apart.
# When creating basis functions, scale should be taken as the diameter of the support size of the input basis functions.
# returns an array containing the indices of the centers, as well as an array of arrays which contain the clustering indices
# list keeps track of which points are still possible choices for the aggregation procedure.
function cluster(centers::AbstractVector{<:SVector}, scale, tree_function, list=falses(length(centers)))
    # aggregation centers
    aggregation_centers = Vector{eltype(centers)}(undef, 0)
    # Id's of the centers selected as aggregation points
    aggregation_indices = Vector{Int}(undef, 0)
    # compute nearby neighbors
    neighborhoods = inrange(tree_function(centers), centers, scale)
    for i = 1 : length(centers)
        # If the element has not been added to the list
        if !list[i]
            push!(aggregation_centers, centers[i])
            push!(aggregation_indices, i)
            # crossing off all nodes that werewithin the neighborhood.
            list[neighborhoods[i]] .= true
        end
    end
    # contains the membership of each element expressed as an integer between 1 and number_of_clusters
    memberships_abstract = nn(tree_function(aggregation_centers), centers)[1]

    # Construct member_lists. Since by definition every aggregation center contains gets at least one member, the following assertion should hold true, and we can discard the aggregation indices returned by the function construct_member_lists
    member_lists = construct_member_lists(memberships_abstract)
    @assert sort(getindex.(member_lists, 1)) == 1 : length(aggregation_indices)
    member_lists = getindex.(member_lists, 2)
    # returns the indices of the domains used as centers of clustering, as well as an array of arrays that contains the members of each cluster
    return aggregation_indices, getindex.(construct_member_lists(memberships_abstract), 2)
end

# Directly compute the new clustered nodes from an input set of clustered nodes.
function cluster(domains::AbstractVector{<:Domain}, scale, tree_function)
    # Where do we start our id's?
    initial_id = maximum(id.(domains)) + 1

    aggregation_centers, aggregation_memberships = cluster(center.(domains), scale, tree_function)
    aggregated_domains = Vector{eltype(domains)}(undef, length(aggregation_centers)) 

    for k = 1 : length(aggregation_centers)
        aggregated_domains[k] = Domain(domains[aggregation_memberships[k]], initial_id + k)
    end
    return aggregated_domains
end

# Function that thakes in a list of points and returns a hierarchical domain decomposition
# centeres contains the point location of the degrees of freedom
# h is the ratio between subsequenct scales,
# centers contains the centers of the degrees of freedom
function create_hierarchy(input_domains::AbstractVector{<:Domain}, h, diams; tree_function=KDTree, h_min = minimum(diams), h_max = max(approximate_diameter(center.(input_domains) / 2), maximum(diams)))
    # the input basis functions should be ordered from coarse to fine, meaning that dims should be sorted in decreasing order.
    @assert issorted(diams, rev=true)
    # Compute the number of levels needed in total
    q = ceil(Int, log(h, h_min / h_max)) + 1
    # vector containing the scales of the different levels
    scales = h_max .* (h .^ (0 : (q - 1)))
    # function that the level to a 
    # We store the original domains that still need to be included into 
    domains_remaining = copy(input_domains)
    # The domains that are passed on from the last level
    domains = Vector{eltype(input_domains)}(undef, 0)
    # Index should figure itself out
    # index = length(centers)
    for k = q : -1 : 1
        # We split the raw domains into domains that are included at the present scale and 
        domains_at_scale = domains_remaining[findall((diams[id.(domains_remaining)] .< (scales[k] / h)))]
        domains_remaining = domains_remaining[findall(.!(diams[id.(domains_remaining)] .< (scales[k] / h)))]
        domains = cluster(vcat(domains, domains_at_scale), scales[k] / h, tree_function)
    end
    return domains 
end

# A function that takes as input the coarsest level of a hierarchical 
# domain decomposition and returns all supernodes on the different levels. Optionally only returns elementary domains
function gather_hierarchy(coarsest::AbstractVector{<:Domain}, elementary=false)
    out = Vector{Vector{eltype(coarsest)}}(undef, 0)

    # recursive function provided with a domain and its input level
    function recurse!(out, in, k::Int)
        # possible increase length of out array
        if length(out) < k
            @assert length(out) == k - 1
            push!(out, Vector{eltype(coarsest)}(undef, 0))
        end 
        push!(out[k], in)
        for child in children(in) 
            recurse!(out, child, k + 1)
        end
    end

    # Initialize recursioin, assigning level 1 to the parent domain
    for coarse_domain in coarsest
        recurse!(out, coarse_domain, 1)
    end

    if elementary
        for k = 1 : length(out)
            out[k] = out[k][findall(iselementary.(out[k]))]
        end
    end
    return out
end

function plot_domains(domains::AbstractVector{<:Domain};
                      xlims=(0.0, 1.0),
                      ylims=(0.0, 1.0),
                      ) 
    @assert length(center(domains[1])) == 2 
    outplot = plot(; xlims, ylims, aspect_ratio=:equal) 
    for domain in domains
        centers = center.(gather_descendants([domain]))
        x = [centers[k][1] for k in 1 : length(centers)]
        y = [centers[k][2] for k in 1 : length(centers)]
        scatter!(outplot, x, y)
    end
    return outplot
end

# # unclear what this is doing :D
# function find_ranges(a::AbstractVector{<:Integer})
#     @assert issorted(a); @assert a[1] == one(eltype(a))
#     n = a[end]
#     indices = Vector{eltype(a)}(undef, n + 1)
#     track_el = zero(eltype(a))
#     for (k, el) in enumerate(a)
#         if el > track_el
#             @assert el == track_el + 1
#             # Note first index of size el
#             indices[el] = k
#             track_el = el
#         end
#     end
#     indices[end] = length(a) + 1
#     return map((x,y) -> x : (y - 1), indices[1 : (end - 1)], indices[2:end])
# end
# 