@testset "construct matrix" begin
    q = 5
    A, coarse_domains, scales, basis_functions, multicolor_ordering, fine_domains = CompressingSolvers.FD_Laplacian_subdivision_2d(q)

    I, J = CompressingSolvers.sparsity_set(multicolor_ordering, CompressingSolvers.center.(fine_domains))
    @test allunique(zip(I, J))
end