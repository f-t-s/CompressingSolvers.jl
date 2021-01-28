using CompressingSolvers
using LinearAlgebra
using Test

@testset "CompressingSolvers.jl" begin
    # Write your tests here.
    @testset "domain.jl" begin
        Δx = 0.05
        Δy = 0.05
        x2d = mapreduce(identity, hcat, [[x; y] for x in 0 : Δx : 1 for y in 0 : Δy : 2])
        diams = 2 * norm([Δx, Δy]) * ones(size(x2d, 2))
        h = 0.5
        domain_vector = CompressingSolvers.array2domains(x2d)
        @test all(CompressingSolvers.iselementary.(domain_vector))
        # Construct a new "parent domain" that contains all children domains
        parent_domain = CompressingSolvers.Domain(domain_vector, length(domain_vector) + 1)
        hierarchy = CompressingSolvers.create_hierarchy(CompressingSolvers.center.(CompressingSolvers.children(parent_domain)), h, diams)
        all_domains = CompressingSolvers.gather_hierarchy(hierarchy)
        @show length(all_domains)
        @show length.(all_domains)
        @show size(x2d)
    end
end