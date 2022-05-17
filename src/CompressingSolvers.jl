module CompressingSolvers
using Base: Order
include("./domains.jl")
include("./basis_functions.jl")
include("./multicolor_ordering.jl")
include("./create_problems.jl")
include("./reconstruction.jl")
include("utils.jl")

# Write your package code here.

# exporting functions 
export uniform2d_fd_poisson
export reconstruct

end
