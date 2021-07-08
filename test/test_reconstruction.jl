@testset "construct matrix" begin
    q = 5
    A, coarse_domains, scales, basis_functions, multicolor_ordering, fine_domains = CompressingSolvers.FD_Laplacian_subdivision_2d(q, 3.0)

    I, J = CompressingSolvers.sparsity_set(multicolor_ordering, CompressingSolvers.center.(fine_domains))
    @test allunique(zip(I, J))

    measurement_results = CompressingSolvers.measure(cholesky(A), multicolor_ordering)

    L = CompressingSolvers.reconstruct(multicolor_ordering, CompressingSolvers.center.(fine_domains), measurement_results)

    @show norm(inv(Matrix(A)) - L * L')
end