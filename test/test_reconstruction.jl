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
        domains, scales, basis_functions = CompressingSolvers.subdivision_2d(2)
        ρ = 3.0
        basis_supernodes, domain_supernodes, multicolor_ordering = CompressingSolvers.supernodal_aggregation_square(domains, scales, basis_functions, ρ)

        𝐅 = CompressingSolvers.SupernodalFactorization(multicolor_ordering, domain_supernodes)

        # setting the nonzero value of 𝐅 to a random buffer
        𝐅.buffer .= rand(length(𝐅.buffer))
        in = CompressingSolvers.SupernodalVector(rand(sum(length.(𝐅.row_supernodes)), 10), 𝐅.row_supernodes)

        # preallocate output
        out = copy(in) 

        # testing the full multiply
        CompressingSolvers.partial_multiply!(out, 𝐅, in)

        L = SparseMatrixCSC(𝐅)
        @test L * L' * Matrix(in) ≈ Matrix(out)
    end

    @testset "reconstruction" begin
        import SparseArrays.sparse 
        # Setting up the test domains
        ρ = 3
        A, domains, scales, basis_functions, basis_supernodes, domain_supernodes, multicolor_ordering = CompressingSolvers.FD_Laplacian_subdivision_2d(3, ρ)

        𝐅 = CompressingSolvers.SupernodalFactorization(multicolor_ordering, domain_supernodes)

        # Ensuring that the lengths of the supernodes are correct
        for (i, j, mat) in zip(findnz(𝐅.data)...)
            @test size(mat) == (length(𝐅.row_supernodes[i]), size(vcat(𝐅.column_supernodes...)[j], 2))
        end

        𝐌 = CompressingSolvers.create_measurement_matrix(multicolor_ordering, 𝐅.row_supernodes)

        𝐎 = CompressingSolvers.measure(inv(Matrix(A)), 𝐌, 𝐅.row_supernodes) 

        CompressingSolvers.reconstruct!(𝐅, 𝐎, multicolor_ordering)
        L = SparseMatrixCSC(𝐅)
    end

end