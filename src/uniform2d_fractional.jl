using FFTW: fft, ifft


# q is the number of levels of the subdivision
# The operator being inverted is u ↦ ((-Δ)²ˢu + βu)
function uniform2d_fractional(q, s, β)
    n = 2 ^ q 
    N = n^2 
    Δx = Δy = 1 / n
    x = 0 : Δx : (1 - Δx) 
    y = 0 : Δy : (1 - Δy)

    @assert length(x) == n
    @assert length(y) == n    # We begin by creating a lower triangular matrix L such that A = L + L'
    lin_inds = LinearIndices((n, n))
    row_inds = Int[]
    col_inds = Int[]
    S = Float64[]
    # contructing the vector that will store the domains
    domains = Vector{Domain{SVector{2, Float64}}}(undef, N)
    for i in 1 : n, j in 1 : n
        domains[lin_inds[i, j]] = Domain(SVector((x[i], y[j])), 
                                         lin_inds[i, j], 
                                         Δx * Δy) 
    end

    # creating a mask to apply in fourier space
    # fourier_mask = zeros(n, n)
    # for i = 1 : n, j = 1 : n
    #     # We need the - 1 due to 1-based indexing.
    #     fourier_mask[i, j] = ((2 * π * (i - 1))^2 + (2 * π * (j - 1))^2)^s + β
    # end

    laplace_mask = zeros(n, n)
    laplace_mask[1, 1] = 2 / Δx^2 + 2 / Δy^2 + β 
    laplace_mask[2, 1] = - 1 / Δx^2
    laplace_mask[n, 1] = -1 / Δx^2
    laplace_mask[1, 2] = - 1 / Δy^2
    laplace_mask[1, n] = -1 / Δy^2
    laplace_mask = fft(laplace_mask)
    zero_order_mask = zeros(n, n)
    zero_order_mask[1, 1] = β



    # the solution oracle
    function ω(A::AbstractVector)
        @assert size(A, 1) == N
        # We unpack the leading dimension of A into the x and y dimension
        unpacked_A = reshape(A, n, n)
        fftA = fft(unpacked_A)    
        fftsol_A = fftA ./ (laplace_mask.^s .+ zero_order_mask)
        solution = ifft(fftsol_A)
        return real.(solution[:])
    end

    function ω(A::AbstractArray)
        @assert size(A, 1) == N
        # We unpack the leading dimension of A into the x and y dimension
        unpacked_A = reshape(A, n, n, size(A)[2:end]...)
        fftA = fft(unpacked_A, (1, 2))    
        fftsol_A = fftA ./ (laplace_mask.^s .+ zero_order_mask)
        solution = ifft(fftsol_A, (1, 2))
        return real.(reshape(solution, N, size(A)[2 : end]...))
    end

    # Returning the problem
    return ReconstructionProblem(domains, PeriodicEuclidean((1.0, 1.0)), ω)
end 

# # q is the number of levels of the subdivision
# # α prescribes the conductivity coefficients by prescribing the edge-weights.
# # It is a function such that α((x + y) / 2) is the conductivity between nodes in position x and y
# # α is only called once on each location, meaning that it can be a random function
# # β is the zeroth order term of the system 
# # β(x) returns the zero order term in the location x.
# function uniform2d_periodic_fd_poisson(q, α = (x, y) -> 1.0, β = (x, y) -> 1.0)
#     # Accounting for periodicity 
#     periodic_α(x, y) = α(div(x, 1), div(y, 1))
#     periodic_β(x, y) = β(div(x, 1), div(y, 1))
#     n = 2 ^ q 
#     N = n^2 
#     Δx = Δy = 1 / n
#     x = 0 : Δx : (1 - Δx) 
#     y = 0 : Δy : (1 - Δy)
#     @assert length(x) == n
#     @assert length(y) == n
#     # We begin by creating a lower triangular matrix L such that A = L + L'
#     lin_inds = LinearIndices((n, n))
#     row_inds = Int[]
#     col_inds = Int[]
#     S = Float64[]
#     # contructing the vector that will store the domains
#     domains = Vector{Domain{SVector{2, Float64}}}(undef, N)
#     for i in 1 : n, j in 1 : n
#         # Construct the Domain and add it to the list
#         domains[lin_inds[i, j]] = 
#             Domain(SVector((x[i], y[j])), 
#                    lin_inds[i, j], 
#                    Δx * Δy) 
#         # adding self-interaction 2
#         α_x = periodic_α(x[i] + Δx, y[j]) / Δx^2
#         α_y = periodic_α(x[i], y[j] + Δy) / Δy^2
#         β_value = periodic_β(x[i], y[j])
# 
#         # Self interaction 
#         push!(col_inds, lin_inds[i, j])
#         push!(row_inds, lin_inds[i, j])
#         push!(S, α_x + α_y + β_value / 2)
# 
#         # Interaction with next point in x direction
#         if i < n 
#             push!(col_inds, lin_inds[i, j])
#             push!(row_inds, lin_inds[i + 1, j])
#             push!(S, - α_x)
#         else
#             push!(col_inds, lin_inds[i, j])
#             push!(row_inds, lin_inds[1, j])
#             push!(S, - α_x)
#         end
# 
# 
#         # Interaction with next point in y direction
#         if j < n 
#             push!(col_inds, lin_inds[i, j])
#             push!(row_inds, lin_inds[i, j + 1])
#             push!(S, - α_y)
#         else
#             push!(col_inds, lin_inds[i, j])
#             push!(row_inds, lin_inds[i, 1])
#             push!(S, - α_y)
#         end
#     end
# 
#     # Assembling the sparse matrix
#     L = sparse(row_inds, col_inds, S)
#     # Forming the full operator by symmetrization.
#     A = L + L'
#     # Returning the problem
#     return ReconstructionProblem(domains, PeriodicEuclidean((1.0, 1.0)), FactorizationOracle(cholesky(A)))
# end 
# 
# 