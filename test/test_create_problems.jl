import LinearAlgebra: norm, I
@testset "Laplacian in two-d" begin
    q = 5
    A, coarse_domains, scales, basis_functions, multicolor_ordering, fine_domains = CompressingSolvers.FD_Laplacian_subdivision_2d(q)
end
