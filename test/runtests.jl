using CompressingSolvers
using LinearAlgebra
using Test
using Plots

@testset "CompressingSolvers.jl" begin
    # Write your tests here.
    @testset "domain.jl in 2d" begin
        Δx = 0.03
        Δy = 0.05
        x = mapreduce(identity, hcat, [[x; y] for x in 0 : Δx : 1 for y in 0 : Δy : 2])
        diams = 2 * norm([Δx, Δy]) * ones(size(x, 2))
        h = 0.5
        domain_vector = CompressingSolvers.array2domains(x)
        @test all(CompressingSolvers.iselementary.(domain_vector))
        # Construct a new "parent domain" that contains all children domains
        parent_domain = CompressingSolvers.Domain(domain_vector, length(domain_vector) + 1)
        # creating the hierarchy from the parent domain
        hierarchy = CompressingSolvers.create_hierarchy((CompressingSolvers.children(parent_domain)), h, diams)
        # returns an array of arrays of domains
        all_domains = CompressingSolvers.gather_hierarchy(hierarchy)
        for k = 1 : length(all_domains)
            sorted_truth = sort(domain_vector, by=CompressingSolvers.id)
            sorted_recovered = sort(CompressingSolvers.gather_descendants(all_domains[k]), by=CompressingSolvers.id)
            
            # Test if the partition, on each level, produces the same set of elements
            @test sort(domain_vector, by=CompressingSolvers.id) == sort(CompressingSolvers.gather_descendants(all_domains[k]), by=CompressingSolvers.id)
        end
    end
    @testset "domain.jl in 3d" begin
        Δx = 0.06
        Δy = 0.07
        Δz = 0.08
        x = mapreduce(identity, hcat, [[x; y; z] for x in 0 : Δx : 1 for y in 0 : Δy : 2 for z in -0.3 : Δz : 0.2])
        diams = 2 * norm([Δx, Δy, Δz]) * ones(size(x, 2))
        h = 0.5
        domain_vector = CompressingSolvers.array2domains(x)
        @test all(CompressingSolvers.iselementary.(domain_vector))
        # Construct a new "parent domain" that contains all children domains
        parent_domain = CompressingSolvers.Domain(domain_vector, length(domain_vector) + 1)
        # creating the hierarchy from the parent domain
        hierarchy = CompressingSolvers.create_hierarchy((CompressingSolvers.children(parent_domain)), h, diams)
        # returns an array of arrays of domains
        all_domains = CompressingSolvers.gather_hierarchy(hierarchy)
        for k = 1 : length(all_domains)
            sorted_truth = sort(domain_vector, by=CompressingSolvers.id)
            sorted_recovered = sort(CompressingSolvers.gather_descendants(all_domains[k]), by=CompressingSolvers.id)
            
            # Test if the partition, on each level, produces the same set of elements
            @test sort(domain_vector, by=CompressingSolvers.id) == sort(CompressingSolvers.gather_descendants(all_domains[k]), by=CompressingSolvers.id)
        end
    end

end