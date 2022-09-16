import LinearAlgebra: norm, I
import Distances: pairwise
@testset "1d Matrix Problem" begin
    N = 5000
    distance = PeriodicEuclidean((N,))
    x = Float64.(Matrix(collect(0 : N-1)'))
    A = exp.(-abs.(pairwise(distance, x, dims=2)) / (N / 10))
    pb = CompressingSolvers.matrix_problem(A, x, distance)
    rk, info = reconstruct(pb, 5.0)
    @test CompressingSolvers.compute_relative_error(rk, pb) ≤ 1e-3
end

@testset "Uniform Laplacian in 2d" begin
    n = 2 ^ 5
    ρ = 7

    pb = uniform2d_dirichlet_fd_poisson(n)
    rk, ~ = reconstruct(pb, ρ)
    @test CompressingSolvers.compute_relative_error(rk, pb) ≤ 1e-4

    pb = uniform2d_neumann_fd_poisson(n)
    rk, ~ = reconstruct(pb, ρ)
    @test CompressingSolvers.compute_relative_error(rk, pb) ≤ 1e-4

    pb = uniform2d_periodic_fd_poisson(n)
    rk, ~ = reconstruct(pb, ρ)
    @test CompressingSolvers.compute_relative_error(rk, pb) ≤ 1e-4
end

@testset "Uniform Laplacian in 3d" begin
    n = 2 ^ 4
    ρ = 4

    pb = uniform3d_dirichlet_fd_poisson(n)
    rk, ~ = reconstruct(pb, ρ)
    @test CompressingSolvers.compute_relative_error(rk, pb) ≤ 5e-3

    pb = uniform3d_neumann_fd_poisson(n)
    rk, ~ = reconstruct(pb, ρ)
    @test CompressingSolvers.compute_relative_error(rk, pb) ≤ 5e-3

    pb = uniform3d_periodic_fd_poisson(n)
    rk, ~ = reconstruct(pb, ρ)
    @test CompressingSolvers.compute_relative_error(rk, pb) ≤ 1e-3
end

@testset "Uniform fractional Laplacian in 2d" begin
    n = 2^5
    ρ = 7

    pb = uniform2d_fractional(n, 0.5, 1.0)
    rk, ~ = reconstruct(pb, ρ)
    @test CompressingSolvers.compute_relative_error(rk, pb) ≤ 1e-4
end

@testset "Uniform fractional Laplacian in 3d" begin
    n = 2 ^ 4
    ρ = 4

    pb = uniform3d_fractional(n, 0.5, 1.0)
    rk, ~ = reconstruct(pb, ρ)
    @test CompressingSolvers.compute_relative_error(rk, pb) ≤ 1e-3
end

@testset "Gridap prolems" begin
    ρ = 6
    pb = gridap_poisson("gridap_models/demo-1.json")
    rk, ~  = reconstruct(pb, ρ)
    @test CompressingSolvers.compute_relative_error(rk, pb) ≤ 1e-3
end