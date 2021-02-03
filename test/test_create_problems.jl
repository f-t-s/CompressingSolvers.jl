import LinearAlgebra: norm, I
@testset "Laplacian in two-d" begin
    q = 5
    A, coarse_domains = CompressingSolvers.FD_Laplacian_subdivision_2d(q)
    all_domains = CompressingSolvers.gather_hierarchy(coarse_domains)
    length.(all_domains) == 4 .^ 1 : q
    basis_vectors, centers = CompressingSolvers.compute_basis_functions(coarse_domains)
    H = hcat(vcat(basis_vectors...)...)
    @test norm(H' * H - I) < 1e-12
end
