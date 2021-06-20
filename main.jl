using LinearAlgebra: Matrix
using CompressingSolvers
using SparseArrays
using LinearAlgebra

# Setting up the test domains
ρ = 6.0
A, domains, scales, basis_functions, basis_supernodes, domain_supernodes, multicolor_ordering, fine_domains = CompressingSolvers.FD_Laplacian_subdivision_2d(6, ρ);

𝐅 = CompressingSolvers.SupernodalFactorization(multicolor_ordering, domain_supernodes);

𝐌 = CompressingSolvers.create_measurement_matrix(multicolor_ordering, 𝐅.row_supernodes);

# 𝐎 = CompressingSolvers.measure(inv(Matrix(A)), 𝐌, 𝐅.row_supernodes); 
@time 𝐎 = CompressingSolvers.measure(cholesky(A), 𝐌, 𝐅.row_supernodes); 

𝐇 = hcat(SparseMatrixCSC.(vcat(𝐅.column_supernodes...))...);

@time CompressingSolvers.reconstruct!(𝐅, 𝐎, 𝐌, multicolor_ordering);
L = SparseMatrixCSC(𝐅)

@show norm(L * L' - inv(Matrix(A))) / norm(inv(Matrix(A)))
@show length(multicolor_ordering)
@show length(reduce(vcat, multicolor_ordering))