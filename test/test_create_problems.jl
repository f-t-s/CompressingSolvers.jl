import LinearAlgebra: norm, I
@testset "Uniform Laplacian in 2d" begin
    q = 5
    ρ = 4

    pb = uniform2d_dirichlet_fd_poisson(q)
    rk = reconstruct(pb, ρ)
    @test CompressingSolvers.compute_relative_error(rk, pb) ≤ 1e-3

    pb = uniform2d_periodic_fd_poisson(q)
    rk = reconstruct(pb, ρ)
    @test CompressingSolvers.compute_relative_error(rk, pb) ≤ 1e-3


end

@testset "Uniform Laplacian in 3d" begin
    q = 4
    ρ = 4

    pb = uniform3d_dirichlet_fd_poisson(q)
    rk = reconstruct(pb, ρ)
    @test CompressingSolvers.compute_relative_error(rk, pb) ≤ 5e-3

    pb = uniform3d_periodic_fd_poisson(q)
    rk = reconstruct(pb, ρ)
    @test CompressingSolvers.compute_relative_error(rk, pb) ≤ 5e-3
end