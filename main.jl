using Random
Random.seed!(172)
using LinearAlgebra: Matrix
using CompressingSolvers
using SparseArrays
using LinearAlgebra

# Setting up the test domains
ρ = 4.0
# ρ = Inf
q = 5
# A, coarse_domains, scales, basis_functions, multicolor_ordering, fine_domains, tree_function = CompressingSolvers.FD_Laplacian_subdivision_2d(q, ρ)
α(x) = rand()
A, coarse_domains, scales, basis_functions, multicolor_ordering, fine_domains, tree_function = CompressingSolvers.FD_periodic_Laplacian_subdivision_2d(q, ρ, α)

measurement_matrix = CompressingSolvers.form_measurement_matrix(multicolor_ordering)
@show size(measurement_matrix)
@time measurement_results = CompressingSolvers.measure(cholesky(A), measurement_matrix)

@time L = CompressingSolvers.reconstruct(multicolor_ordering, CompressingSolvers.center.(fine_domains), measurement_matrix, measurement_results, tree_function)

# @show opnorm(inv(Matrix(A)) - L * L') / opnorm(inv(Matrix(A)))
@show CompressingSolvers.compute_relative_error(L, cholesky(A), 200, 200)
# 𝐅 = CompressingSolvers.SupernodalFactorization(multicolor_ordering, domain_supernodes);
# 
# 𝐌 = CompressingSolvers.create_measurement_matrix(multicolor_ordering, 𝐅.row_supernodes);
# 
# # 𝐎 = CompressingSolvers.measure(inv(Matrix(A)), 𝐌, 𝐅.row_supernodes); 
# @time 𝐎 = CompressingSolvers.measure(cholesky(A), 𝐌, 𝐅.row_supernodes); 
# 
# 𝐇 = hcat(SparseMatrixCSC.(vcat(𝐅.column_supernodes...))...);
# 
# @time CompressingSolvers.reconstruct!(𝐅, 𝐎, 𝐌, multicolor_ordering);
# L = SparseMatrixCSC(𝐅)
# 
# # @show norm(L * L' - inv(Matrix(A))) / norm(inv(Matrix(A)))
# @show length(multicolor_ordering)
# @show length(reduce(vcat, multicolor_ordering))
# @show CompressingSolvers.compute_relative_error(L, cholesky(A))