@testset "construct matrix" begin
    q = 5
    A, coarse_domains, scales, basis_functions, multicolor_ordering, fine_domains, tree_function = CompressingSolvers.FD_Laplacian_subdivision_2d(q, 4.0)

    I, J = CompressingSolvers.sparsity_set(multicolor_ordering, CompressingSolvers.center.(fine_domains), tree_function)
    @test allunique(zip(I, J))

    measurement_matrix = CompressingSolvers.form_measurement_matrix(multicolor_ordering)
    measurement_results = CompressingSolvers.measure(cholesky(A), measurement_matrix)


    L = CompressingSolvers.reconstruct(multicolor_ordering, CompressingSolvers.center.(fine_domains), measurement_matrix, measurement_results, tree_function)

    @show norm(inv(Matrix(A)) - L * L')
end