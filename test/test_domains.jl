using NearestNeighbors: KDTree
@testset "in 2d" begin
    h = 0.3
    Δx = 0.03
    Δy = 0.05
    x = mapreduce(identity, hcat, [[x; y] for x in 0 : Δx : 1 for y in 0 : Δy : 2])
    diams = 2 * norm([Δx, Δy]) * ones(size(x, 2))
    domain_vector = CompressingSolvers.array2domains(x)
    @test all(CompressingSolvers.iselementary.(domain_vector))
    # Construct a new "parent domain" that contains all children domains
    parent_domain = CompressingSolvers.Domain(domain_vector, length(domain_vector) + 1)
    # creating the hierarchy from the parent domain
    hierarchy = CompressingSolvers.create_hierarchy((CompressingSolvers.children(parent_domain)), h, diams, KDTree)
    # returns an array of arrays of domains
    all_domains = CompressingSolvers.gather_hierarchy(hierarchy)
    for k = 1 : length(all_domains)
        sorted_truth = sort(domain_vector, by=CompressingSolvers.id)
        sorted_recovered = sort(CompressingSolvers.gather_descendants(all_domains[k]), by=CompressingSolvers.id)

        # make sure the weights stay the same
        @test sum(CompressingSolvers.weight.(all_domains[k])) == sum(CompressingSolvers.weight.(domain_vector))

        # Test if the partition, on each level, produces the same set of elements
        @test sort(domain_vector, by=CompressingSolvers.id) == sort(CompressingSolvers.gather_descendants(all_domains[k]), by=CompressingSolvers.id)
    end
    CompressingSolvers.plot_domains(all_domains[1]; xlims=(0.0, 2.0), ylims=(0.0,2.0))
end

@testset "in 2d, inhomogeneous" begin
    h = 0.3
    Δratio = 3
    Δx = 0.03
    Δy = 0.05
    x1 = mapreduce(identity, hcat, [[x; y] for x in 0 : Δx : 1 for y in 0 : Δy : 2])
    x2 = mapreduce(identity, hcat, [[x; y] for x in (1 + Δx * Δratio)  : (Δx * Δratio) : 2 for y in (Δy * Δratio): (Δy * Δratio) : 1])
    diams = vcat(2 * norm([Δx, Δy]) * ones(size(x2, 2)) * Δratio, 2 * norm([Δx, Δy]) * ones(size(x1, 2)))
    x = hcat(x2, x1)
    domain_vector = CompressingSolvers.array2domains(x, vcat(Δratio^2 * ones(size(x2, 2)), ones(size(x1, 2))))
    @test all(CompressingSolvers.iselementary.(domain_vector))
    # Construct a new "parent domain" that contains all children domains
    parent_domain = CompressingSolvers.Domain(domain_vector, length(domain_vector) + 1)
    # creating the hierarchy from the parent domain
    hierarchy = CompressingSolvers.create_hierarchy((CompressingSolvers.children(parent_domain)), h, diams, KDTree)
    # returns an array of arrays of domains
    all_domains = CompressingSolvers.gather_hierarchy(hierarchy)

    # Test whether there are no "spurious scales"
    @test length(unique(length.(all_domains))) == length(all_domains) 
    for k = 1 : length(all_domains)
        sorted_truth = sort(domain_vector, by=CompressingSolvers.id)
        sorted_recovered = sort(CompressingSolvers.gather_descendants(all_domains[k]), by=CompressingSolvers.id)

        if k < 3
            # make sure the weights stay the same
            @test sum(CompressingSolvers.weight.(all_domains[k])) == sum(CompressingSolvers.weight.(domain_vector))

            # Test if the partition, on each level, produces the same set of elements
            @test sort(domain_vector, by=CompressingSolvers.id) == sort(CompressingSolvers.gather_descendants(all_domains[k]), by=CompressingSolvers.id)
        end
    end
    CompressingSolvers.plot_domains(all_domains[1]; xlims=(0.0, 2.0), ylims=(0.0,2.0))
end

@testset "in 3d" begin
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
    hierarchy = CompressingSolvers.create_hierarchy((CompressingSolvers.children(parent_domain)), h, diams, KDTree)
    # returns an array of arrays of domains
    all_domains = CompressingSolvers.gather_hierarchy(hierarchy)
    # Test whether there are no "spurious scales"
    @test length(unique(length.(all_domains))) == length(all_domains) 
    for k = 1 : length(all_domains)
        sorted_truth = sort(domain_vector, by=CompressingSolvers.id)
        sorted_recovered = sort(CompressingSolvers.gather_descendants(all_domains[k]), by=CompressingSolvers.id)

        # Test if the partition, on each level, produces the same set of elements
        @test sort(domain_vector, by=CompressingSolvers.id) == sort(CompressingSolvers.gather_descendants(all_domains[k]), by=CompressingSolvers.id)
    end
end