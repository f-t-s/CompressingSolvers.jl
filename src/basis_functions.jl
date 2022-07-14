import SparseArrays: SparseVector, sparsevec
import LinearAlgebra: cholesky, Matrix

struct BasisFunction{PT,RT}
    center::PT
    coefficients::SparseVector{RT, Int}
end

function BasisFunction(center, coefficients)
    return BasisFunction{typeof(center), eltype(coefficients)}(center, coefficients) 
end

function real_type(::Type{BasisFunction{PT, RT}})  where {PT, RT}
    return RT
end

function center(bf::BasisFunction)
    return bf.center
end

function coefficients(bf::BasisFunction)
    return bf.coefficients
end

function create_basis_vector(id_arrays, weights, N)
    @assert size(id_arrays) == size(weights)
    out_weights = vcat([weights[k] * one.(id_arrays[k]) for k = 1 : length(weights)]...)
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

            # Compute coefficients of linear combination
            M = zeros(eltype(PT), n_ch + 1, n_ch + 1)
            M[1, 1] = sum(weight.(children(dm)))
            # setting the inner products.
            for i = 1 : n_ch; 
                M[1, i + 1] = M[i + 1, 1] = M[i + 1, i + 1] = weight(children(dm)[i]) 
            end
            # Adding 1 to the last entry of M to ensure positivity# 
            # This should not affect the result, since the affected columns are dropped
            M[end, end] += 1
            # Compute QR factorization that will provide coefficients 
            R = inv(Matrix(cholesky(M).L'))

            # apply the function to the children to obtain the children arrays
            id_arrays = map(ch -> recurse(ch, k + 1), children(dm))
            id_arrays = vcat([vcat(id_arrays...)], id_arrays)

            # If on first level, return normalized average as basis function

            if k == 1
                push!(basis_vectors[k], create_basis_vector(id_arrays, R[:, 1], N))
                push!(centers[k], center(dm))
            end

            # add the remaining basis functions
            for ch = 2 : n_ch
                push!(basis_vectors[k + 1], create_basis_vector(id_arrays, R[:, ch], N))
                push!(centers[k + 1], center(dm))
            end
            return id_arrays[1]
        end
    end
    for dm in domains
        recurse(dm, 1)
    end

    # combine basis_vectors and centers to basis functions
    # cannot broadcast directly, because we are dealing with arrays of arrays of basis functions due to the different scales.
    out = Vector{Vector{BasisFunction{eltype(centers[1]), eltype(basis_vectors[1][1])}}}(undef, length(basis_vectors))
    for k = 1 : length(basis_vectors)
        out[k] = BasisFunction.(centers[k], basis_vectors[k])
    end

    return out
end