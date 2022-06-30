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

include("uniform2d_poisson.jl")
include("uniform3d_poisson.jl")
include("uniform2d_fractional.jl")
include("uniform3d_fractional.jl")

