import SparseArrays.SparseMatrixCSC
import SparseArrays: findnz, dropzeros!, spdiagm
import StaticArrays: SVector
import Base.*
using NearestNeighbors: BallTree, KDTree
using Distances: Euclidean, PeriodicEuclidean

struct FactorizationOracle
    factorization
end

# TODO: Make FactorizationOracle a function-like object
function (fct_oracle::FactorizationOracle)(v)
    return fct_oracle.factorization \ v
end

struct ReconstructionProblem
    # Vector of elementary domains 
    domains
    # Distance used to compute clustering.
    # Will be either PeriodicEuclidean or Euclidean for most applications  
    distance
    # A function that is able to compute the matrix-vector products for m 
    # right hand sides, passed as a N × m matrix
    ω
end

function *(pb::ReconstructionProblem, v)
    return pb.ω(v)
end

# q is the number of levels of the subdivision
# α prescribes the conductivity coefficients by prescribing the edge-weights.
# It is a function such that α((x + y) / 2) is the conductivity between nodes in position x and y
# α is only called once on each location, meaning that it can be a random function
# β is the zeroth order term of the system 
# β(x) returns the zero order term in the location x.
function uniform2d_fd_poisson(q, α = (x, y) -> 1.0, β = (x, y) -> 0.0)
    n = 2 ^ q 
    N = n^2 
    Δx = Δy = 1 / (n + 1)
    x = Δx : Δx : (1 - Δx)
    y = Δy : Δy : (1 - Δy)
    # We begin by creating a lower triangular matrix L such that A = L + L'
    lin_inds = LinearIndices((n, n))
    row_inds = Int[]
    col_inds = Int[]
    S = Float64[]
    # contructing the vector that will store the domains
    domains = Vector{Domain{SVector{2, Float64}}}(undef, N)
    for i in 1 : n, j in 1 : n
        # Construct the Domain and add it to the list
        domains[lin_inds[i, j]] = 
            Domain(SVector((x[i], y[j])), 
                   lin_inds[i, j], 
                   Δx * Δy) 
        # adding self-interaction 2
        α_x = α(x[i] + Δx, y[j])
        α_y = α(x[i], y[j] + Δy)
        β_value = β(x[i], y[j])

        # Self interaction 
        push!(col_inds, lin_inds[i, j])
        push!(row_inds, lin_inds[i, j])
        push!(S, α_x + α_y + β_value / 2)

        # Interaction with next point in x direction
        if i < n 
            push!(col_inds, lin_inds[i, j])
            push!(row_inds, lin_inds[i + 1, j])
            push!(S, - α_x)
        end

        # Interaction with next point in y direction
        if j < n 
            push!(col_inds, lin_inds[i, j])
            push!(row_inds, lin_inds[i, j + 1])
            push!(S, - α_y)
        end
    end

    # Assembling the sparse matrix
    L = sparse(row_inds, col_inds, S)
    # Forming the full operator by symmetrization.
    A = L + L'
    # Returning the problem
    return ReconstructionProblem(domains, Euclidean(), FactorizationOracle(cholesky(A)))
end 