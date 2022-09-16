@testset "on dirichlet conditions" begin
    n = 2 ^ 5
    h = 0.5
    ρ = 5.0
    pb = uniform2d_dirichlet_fd_poisson(n)

    # This is a part of the reconstruct(::ReconstructionProblem, ρ, h=0.5) wrapper function
    ##########
    TreeType = KDTree
    tree_function(x) = TreeType(x, pb.distance)
    domain_hierarchy = CompressingSolvers.gather_hierarchy(CompressingSolvers.create_hierarchy(pb.domains, h, KDTree))
    scales = [maximum(CompressingSolvers.approximate_scale(CompressingSolvers.center.(domain_hierarchy[k]), tree_function)) for k = 1 : length(domain_hierarchy)]
    basis_functions = CompressingSolvers.compute_basis_functions(first(domain_hierarchy)) 
    multicolor_ordering = CompressingSolvers.construct_multicolor_ordering(basis_functions, ρ * scales, tree_function)
    ##########

    I, J = CompressingSolvers.sparsity_set(multicolor_ordering, CompressingSolvers.center.(pb.domains), tree_function)
    @test allunique(zip(I, J))
end

@testset "on periodic conditions" begin
    n = 2 ^ 5
    h = 0.5 
    ρ = 5.0
    pb = uniform2d_periodic_fd_poisson(n)

    # This is a part of the reconstruct(::ReconstructionProblem, ρ, h=0.5) wrapper function
    ##########
    TreeType = BallTree
    tree_function(x) = TreeType(x, pb.distance)
    domain_hierarchy = CompressingSolvers.gather_hierarchy(CompressingSolvers.create_hierarchy(pb.domains, h, TreeType))
    scales = [maximum(CompressingSolvers.approximate_scale(CompressingSolvers.center.(domain_hierarchy[k]), tree_function)) for k = 1 : length(domain_hierarchy)]
    basis_functions = CompressingSolvers.compute_basis_functions(first(domain_hierarchy)) 
    multicolor_ordering = CompressingSolvers.construct_multicolor_ordering(basis_functions, ρ * scales, tree_function)
    ##########

    I, J = CompressingSolvers.sparsity_set(multicolor_ordering, CompressingSolvers.center.(pb.domains), tree_function)
    @test allunique(zip(I, J))
end