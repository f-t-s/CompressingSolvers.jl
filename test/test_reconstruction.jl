@testset "data structures" begin
    import Random.randperm
    M = 21
    N = 10
    rp = randperm(M)
    row_supernodes = [[rp[1:3]]; [rp[4:8]]; [rp[9:10]]; [rp[11:17]]; [rp[18:21]]]
    𝐌 = rand(M, N)
    super_𝐌 = CompressingSolvers.SupernodalColumn(𝐌, row_supernodes)
    @test 𝐌 == Matrix(super_𝐌)
end