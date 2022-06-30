using Random
Random.seed!(172)
using LinearAlgebra: Matrix
using CompressingSolvers
using SparseArrays
using LinearAlgebra
using GLMakie

# Setting up the test domains
ρ = 6
# ρ = Inf
q = 7


# pb = uniform2d_dirichlet_fd_poisson(q)
# pb = uniform3d_dirichlet_fd_poisson(q)

pb = uniform2d_fractional(q, 0.50, 1.0)
# pb = uniform2d_periodic_fd_poisson(q)
# pb = uniform2d_dirichlet_fd_poisson(q)

rk = reconstruct(pb, ρ) 

# # Making sure the unpacking-packing doesn't go awry.
# a = rand(4^q)
# b = rand(4^q)
# @show norm(hcat(pb * a, pb * b) - pb * hcat(a, b))

# n = 2 ^ q; N = n^2 
# h = 1 / n
# x = 0 : h : (1 - h) 
# y = 0 : h : (1 - h)
# rhs = rand(N)
# wireframe(x, y, 1000 * reshape(pb1 * rhs, n, n))
# wireframe(x, y, 400 * reshape(pb2 * rhs, n, n))
# wireframe(x, y, 400 * reshape(rk * rhs, n, n))

# @show norm(pb1 * rhs - pb2 * rhs) / norm(pb1 * rhs)
# @show norm((pb1 * rhs) / (pb1 * rhs) - (pb2 * rhs) / (pb2 * rhs)) / norm(pb1 * rhs)

CompressingSolvers.compute_relative_error(rk, pb)