using Random
Random.seed!(172)
using LinearAlgebra: Matrix
using CompressingSolvers
using SparseArrays
using LinearAlgebra

# # Setting up the test domains
# ρ = 3
# # ρ = Inf
# q = 4
# 
# 
# # pb = uniform2d_dirichlet_fd_poisson(q)
# # pb = uniform3d_dirichlet_fd_poisson(q)
# 
# pb = uniform2d_periodic_fd_poisson(q)
# # pb = uniform2d_dirichlet_fd_poisson(q)
# 
# rk = reconstruct(pb, ρ) 
# 
# CompressingSolvers.compute_relative_error(rk, pb)

q = 5
h = 0.5 
ρ = 5.0
pb = uniform2d_periodic_fd_poisson(q)

# This is a part of the reconstruct(::ReconstructionProblem, ρ, h=0.5) wrapper function
##########
pb.domains .= pb.domains[randperm(length(pb.domains))]
TreeType = BallTree
tree_function(x) = TreeType(x, pb.distance)
# tree_function(x) = TreeType(x, Euclidean())
domain_hierarchy = CompressingSolvers.gather_hierarchy(CompressingSolvers.create_hierarchy(pb.domains, h, TreeType))
aggregation_centers, aggregation_indices = CompressingSolvers.compute_aggregation_centers(CompressingSolvers.center.(pb.domains), 0.05, tree_function)
clustered_domains = CompressingSolvers.cluster(pb.domains, 0.07125, tree_function) 
scales = [maximum(CompressingSolvers.approximate_scale(CompressingSolvers.center.(domain_hierarchy[k]), tree_function)) for k = 1 : length(domain_hierarchy)]
basis_functions = CompressingSolvers.compute_basis_functions(first(domain_hierarchy)) 
multicolor_ordering = CompressingSolvers.construct_multicolor_ordering(basis_functions, ρ * scales, tree_function)
##########

I, J = CompressingSolvers.sparsity_set(multicolor_ordering, CompressingSolvers.center.(pb.domains), tree_function)
