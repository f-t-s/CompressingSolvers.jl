import SparseArrays: SparseVector, sparsevec
import LinearAlgebra: qr, Matrix

function create_basis_vector(id_arrays, weights, N)
    @assert size(id_arrays) == size(weights)
    normalization_constant = sqrt(sum((weights.^2).*(length.(id_arrays))))
    @show typeof.(id_arrays)
    out_weights = vcat([weights[k] * one.(id_arrays[k]) for k = 1 : length(weights)]...) / normalization_constant
    return sparsevec(vcat(id_arrays...), out_weights, N)
end

# Input are the coarsest-scale domains
function compute_basis_functions(domains::AbstractVector{Domain{PT}}, N = maximum(id.(gather_descendants(domains)))) where PT<:AbstractVector{<:Real}
    centers = Vector{Vector{PT}}(undef,1) 
    centers[1] = Vector{PT}(undef, 0)
    basis_vectors = Vector{Vector{SparseVector{eltype(PT), Int}}}(undef,1) 
    basis_vectors[1] = Vector{SparseVector{eltype(PT), Int}}(undef, 0)
    # recursive functions that computes the basis functions orthogonal in ℓ^2 inner product. Here, k is the level of domain
    function recurse(dm, k)
        # If the domain is elementrary, we return an array containint its index.
        println("Entering recurse")
        n_ch = length(children(dm))
        if iselementary(dm)
            return [id(dm)]
        else
            # Extend length of output arrays if first domain on new level
            if k + 1 > length(basis_vectors) 
                # make sure we didn't skip a level
                @assert length(basis_vectors) == length(centers) == k
                push!(centers, Vector{PT}(undef, 0))
                push!(basis_vectors, Vector{SparseVector{eltype(PT), Int}}(undef, 0))
            end 

            # TODO: Fix orthogonalization procedure!
            # Compute coefficients of linear combination
            M = zeros(eltype(PT), n_ch, n_ch + 1)
            M[: , 1] .= one(eltype(M))
            # setting the entries of M to reproduce inner products
            for i = 1 : n_ch; M[i, i + 1] = weight(children(dm)[i]); end
            # Compute QR factorization that will provide coefficients 
            R = inv(Matrix(qr(M).R)[:, 1 : n_ch])

            # apply the function to the children to obtain the children arrays
            id_arrays = map(ch -> recurse(ch, k + 1), children(dm))

            # If on first level, return normalized average as basis function

            @show k
            if k == 1
                push!(basis_vectors[k], create_basis_vector(id_arrays, R[:, 1], N))
                push!(centers[k], center(dm))
            end

            # add the remaining basis functions
            for ch = 2 : n_ch
                push!(basis_vectors[k + 1], create_basis_vector(id_arrays, R[:, ch], N))
                push!(centers[k + 1], center(dm))
            end
            return vcat(id_arrays...)
        end
    end
    for dm in domains
        recurse(dm, 1)
    end

    return basis_vectors, centers 
end