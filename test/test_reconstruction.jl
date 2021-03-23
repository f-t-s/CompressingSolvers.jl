@testset "data structures" begin
    @testset "SupernodalVector" begin
        import Random.randperm
        M = 21
        N = 10
        rp = randperm(M)
        row_supernodes = [[rp[1:3]]; [rp[4:8]]; [rp[9:10]]; [rp[11:17]]; [rp[18:21]]]
        𝐌 = rand(M, N)
        super_𝐌 = CompressingSolvers.SupernodalVector(𝐌, row_supernodes)
        @test 𝐌 == Matrix(super_𝐌)
    end

    @testset "SupernodalSparseVector" begin
        import Random.randperm
        import SparseArrays: sprand, SparseMatrixCSC, findnz
        M = 21
        N = 10
        rp = randperm(M)
        row_supernodes = [[rp[1:2]]; [rp[3:7]]; [rp[8:14]]; [rp[15:17]]; [rp[18:21]]]
        𝐌 = sprand(M, N, 0.02)
        super_𝐌 = CompressingSolvers.SupernodalSparseVector(𝐌, row_supernodes)
        @test 𝐌 == SparseMatrixCSC(super_𝐌)
    end

    @testset "SupernodalFactorization" begin
        # Setting up the test domains
        domains, scales, basis_functions = CompressingSolvers.subdivision_2d(5)
        ρ = 3.0
        basis_supernodes, domain_supernodes, multicolor_ordering = CompressingSolvers.supernodal_aggregation_square(domains, scales, basis_functions, ρ)

        CompressingSolvers.SupernodalFactorization(multicolor_ordering, domain_supernodes)
    end

    @testset "reconstruction" begin
        # Setting up the test domains
        ρ = 3.0
        A, domains, scales, basis_functions, basis_supernodes, domain_supernodes, multicolor_ordering = CompressingSolvers.FD_Laplacian_subdivision_2d(3, ρ)

        𝐅 = CompressingSolvers.SupernodalFactorization(multicolor_ordering, domain_supernodes)

        𝐌 = CompressingSolvers.create_measurement_matrix(multicolor_ordering, 𝐅.row_supernodes)

        CompressingSolvers.measure(inv(Matrix(A)), 𝐌, 𝐅.row_supernodes) 
    end

end