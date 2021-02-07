import LinearAlgebra: norm, I
@testset "Laplacian in two-d" begin
    q = 5
    A, coarse_domains, scales, basis_functions, basis_supernodes, domain_supernodes, multicolor_ordering = CompressingSolvers.FD_Laplacian_subdivision_2d(q)
    all_domains = CompressingSolvers.gather_hierarchy(coarse_domains)
    @test length.(all_domains) == 4 .^ (1 : q)
    H = hcat(CompressingSolvers.coefficients.(vcat(basis_functions...))...)
    @test norm(H' * H - I) < 1e-12
end
