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
export uniform2d_dirichlet_fd_poisson
export uniform2d_neumann_fd_poisson
export uniform2d_periodic_fd_poisson
export uniform2d_fractional
export uniform3d_dirichlet_fd_poisson
export uniform3d_neumann_fd_poisson
export uniform3d_periodic_fd_poisson
export uniform3d_fractional
export gridap_poisson
export reconstruct

end
