using CompressingSolvers
using SparseArrays
using LinearAlgebra

# Setting up the test domains
ρ = 3.0
A, domains, scales, basis_functions, basis_supernodes, domain_supernodes, multicolor_ordering = CompressingSolvers.FD_Laplacian_subdivision_2d(5, ρ);

𝐅 = CompressingSolvers.SupernodalFactorization(multicolor_ordering, domain_supernodes);

𝐌 = CompressingSolvers.create_measurement_matrix(multicolor_ordering, 𝐅.row_supernodes);

𝐎 = CompressingSolvers.measure(inv(Matrix(A)), 𝐌, 𝐅.row_supernodes); 

𝐇 = hcat(SparseMatrixCSC.(vcat(𝐅.column_supernodes...))...);

CompressingSolvers.reconstruct!(𝐅, 𝐎, 𝐌, multicolor_ordering);
L = SparseMatrixCSC(𝐅)

@show norm(L * L' - inv(Matrix(A)))