
# q is the number of levels of the subdivision
# The operator being inverted is u ↦ ((-Δ)²ˢu + βu)
function uniform3d_fractional(q, s, β)
    n = 2 ^ q 
    N = n^3 
    Δx = Δy = Δz = 1 / n
    x = 0 : Δx : (1 - Δx)
    y = 0 : Δy : (1 - Δy)
    z = 0 : Δz : (1 - Δz)
    @assert length(x) == n
    @assert length(y) == n
    @assert length(z) == n
    # We begin by creating a lower triangular matrix L such that A = L + L'
    lin_inds = LinearIndices((n, n, n))
        # contructing the vector that will store the domains
    domains = Vector{Domain{SVector{3, Float64}}}(undef, N)
    for i in 1 : n, j in 1 : n, k in 1 : n
        domains[lin_inds[i, j, k]] = Domain(SVector((x[i], y[j], z[k])), 
                                         lin_inds[i, j, k], 
                                         Δx * Δy * Δz) 
    end

    laplace_mask = zeros(n, n, n)
    laplace_mask[1, 1, 1] = 2 / Δx^2 + 2 / Δy^2 + 2 / Δz^2 + β 
    laplace_mask[2, 1, 1] = - 1 / Δx^2
    laplace_mask[n, 1, 1] = -1 / Δx^2
    laplace_mask[1, 2, 1] = - 1 / Δy^2
    laplace_mask[1, n, 1] = -1 / Δy^2
    laplace_mask[1, 1, 2] = - 1 / Δz^2
    laplace_mask[1, 1, n] = -1 / Δz^2
    laplace_mask = fft(laplace_mask)
    zero_order_mask = zeros(n, n, n)
    zero_order_mask[1, 1, 1] = β

    # the solution oracle
    function ω(A::AbstractVector)
        @assert size(A, 1) == N
        # We unpack the leading dimension of A into the x and y dimension
        unpacked_A = reshape(A, n, n, n)
        fftA = fft(unpacked_A)    
        fftsol_A = fftA ./ (laplace_mask.^s .+ zero_order_mask)
        solution = ifft(fftsol_A)
        return real.(solution[:])
    end

    function ω(A::AbstractArray)
        @assert size(A, 1) == N
        # We unpack the leading dimension of A into the x and y dimension
        unpacked_A = reshape(A, n, n, n, size(A)[2:end]...)
        fftA = fft(unpacked_A, (1, 2, 3))    
        fftsol_A = fftA ./ (laplace_mask.^s .+ zero_order_mask)
        solution = ifft(fftsol_A, (1, 2, 3))
        return real.(reshape(solution, N, size(A)[2 : end]...))
    end

    # Returning the problem
    return ReconstructionProblem(domains, PeriodicEuclidean((1.0, 1.0, 1.0)), ω)
end 