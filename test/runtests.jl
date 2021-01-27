using CompressingSolvers
using Test

@testset "CompressingSolvers.jl" begin
    # Write your tests here.
    @testset "domain.jl" begin
        x2d = rand(2, 2000)            
        domain_vector = CompressingSolvers.array2domains(x2d)
        @test all(CompressingSolvers.iselementary.(domain_vector))
        # Construct a new "parent domain" that contains all children domains
        parent_domain = CompressingSolvers.Domain(domain_vector, length(domain_vector) + 1)
        # hierarchy = CompressingSolvers.create_hierarchy(parent_domain
    end
end
