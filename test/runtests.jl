using CompressingSolvers
using LinearAlgebra
using Test
using Plots

@testset "CompressingSolvers.jl" begin
    # Write your tests here.
    # Testing domain.jl
    @testset "domain.jl" begin
        include("test_domains.jl")
    end
    # Testing creating_problems.jl
    @testset "domain.jl" begin
        include("./test_create_problems.jl")
    end
end