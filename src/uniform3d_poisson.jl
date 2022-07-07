# q is the number of levels of the subdivision
# α prescribes the conductivity coefficients by prescribing the edge-weights.
# It is a function such that α((x + y) / 2) is the conductivity between nodes in position x and y
# α is only called once on each location, meaning that it can be a random function
# β is the zeroth order term of the system 
# β(x) returns the zero order term in the location x.
function uniform3d_dirichlet_fd_poisson(q, α = (x, y, z) -> 1.0, β = (x, y, z) -> 0.0)
    n = 2 ^ q 
    N = n^3 
    Δx = Δy = Δz = 1 / (n + 1)
    x = Δx : Δx : (1 - Δx)
    y = Δy : Δy : (1 - Δy)
    z = Δz : Δz : (1 - Δz)
    @assert length(x) == n
    @assert length(y) == n
    @assert length(z) == n
    # We begin by creating a lower triangular matrix L such that A = L + L'
    lin_inds = LinearIndices((n, n, n))
    row_inds = Int[]
    col_inds = Int[]
    S = Float64[]
    # contructing the vector that will store the domains
    domains = Vector{Domain{SVector{3, Float64}}}(undef, N)
    for i in 1 : n, j in 1 : n, k in 1 : n
        # Construct the Domain and add it to the list
        domains[lin_inds[i, j, k]] = 
            Domain(SVector((x[i], y[j], z[k])), 
                   lin_inds[i, j, k], 
                   Δx * Δy * Δz) 
        # adding self-interaction 2
        α_x = α(x[i] + Δx / 2, y[j], z[k]) / Δx^2
        α_y = α(x[i], y[j] + Δy / 2, z[k]) / Δy^2
        α_z = α(x[i], y[j], z[k] + Δz / 2) / Δz^2
        β_value = β(x[i], y[j], z[k])

        # Self interaction 
        push!(col_inds, lin_inds[i, j, k])
        push!(row_inds, lin_inds[i, j, k])
        push!(S, β_value)

        if i == 1
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, α(x[i] - Δx / 2, y[j], z[k]) / Δx^2)
        end

        # Interaction with next point in x direction
        if i < n 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i + 1, j, k])
            push!(S, - α_x)

            push!(col_inds, lin_inds[i + 1, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, - α_x)

            # Self interaction 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, α_x)

            # Self interaction 
            push!(col_inds, lin_inds[i + 1, j, k])
            push!(row_inds, lin_inds[i + 1, j, k])
            push!(S, α_x)
        else
            # Self interaction 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, α_x)
        end

        # substitute for the missing dof at j=0
        if j == 1 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, α(x[i], y[j] - Δy / 2, z[k]) / Δy^2)
        end

        # Interaction with next point in y direction
        if j < n 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j + 1, k])
            push!(S, - α_y)

            push!(col_inds, lin_inds[i, j + 1, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, - α_y)

            # Self interaction 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, α_y)

            # Self interaction 
            push!(col_inds, lin_inds[i, j + 1, k])
            push!(row_inds, lin_inds[i, j + 1, k])
            push!(S, α_y)
        else 
            # Self interaction 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, α_y)
        end
    
        # substitute for the missing dof at k=0
        if k == 1 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, α(x[i], y[j], z[k] - Δz / 2) / Δz^2)
        end

        # Interaction with next point in z direction
        if k < n 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k + 1])
            push!(S, - α_z)

            push!(col_inds, lin_inds[i, j, k + 1])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, - α_z)

            # Self interaction 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, α_z)

            # Self interaction 
            push!(col_inds, lin_inds[i, j, k + 1])
            push!(row_inds, lin_inds[i, j, k + 1])
            push!(S, α_z)
        else 
            # Self interaction 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, α_z)
        end
    end

    # Assembling the sparse matrix
    A = sparse(row_inds, col_inds, S)
    # Forming the full operator by symmetrization.
    # Returning the problem
    return ReconstructionProblem(domains, Euclidean(), FactorizationOracle(cholesky(A)))
end 

# q is the number of levels of the subdivision
# α prescribes the conductivity coefficients by prescribing the edge-weights.
# It is a function such that α((x + y) / 2) is the conductivity between nodes in position x and y
# α is only called once on each location, meaning that it can be a random function
# β is the zeroth order term of the system 
# β(x) returns the zero order term in the location x.
function uniform3d_neumann_fd_poisson(q, α = (x, y, z) -> 1.0, β = (x, y, z) -> 1.0)
    n = 2 ^ q 
    N = n^3 
    Δx = Δy = Δz = 1 / (n + 1)
    x = Δx : Δx : (1 - Δx)
    y = Δy : Δy : (1 - Δy)
    z = Δz : Δz : (1 - Δz)
    @assert length(x) == n
    @assert length(y) == n
    @assert length(z) == n
    # We begin by creating a lower triangular matrix L such that A = L + L'
    lin_inds = LinearIndices((n, n, n))
    row_inds = Int[]
    col_inds = Int[]
    S = Float64[]
    # contructing the vector that will store the domains
    domains = Vector{Domain{SVector{3, Float64}}}(undef, N)
    for i in 1 : n, j in 1 : n, k in 1 : n
        # Construct the Domain and add it to the list
        domains[lin_inds[i, j, k]] = 
            Domain(SVector((x[i], y[j], z[k])), 
                   lin_inds[i, j, k], 
                   Δx * Δy * Δz) 
        # adding self-interaction 2
        α_x = α(x[i] + Δx / 2, y[j], z[k]) / Δx^2
        α_y = α(x[i], y[j] + Δy / 2, z[k]) / Δy^2
        α_z = α(x[i], y[j], z[k] + Δz / 2) / Δz^2
        β_value = β(x[i], y[j], z[k])

        # Self interaction 
        push!(col_inds, lin_inds[i, j, k])
        push!(row_inds, lin_inds[i, j, k])
        push!(S, β_value)

        # Interaction with next point in x direction
        if i < n 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i + 1, j, k])
            push!(S, - α_x)

            push!(col_inds, lin_inds[i + 1, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, - α_x)

            # Self interaction 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, α_x)

            # Self interaction 
            push!(col_inds, lin_inds[i + 1, j, k])
            push!(row_inds, lin_inds[i + 1, j, k])
            push!(S, α_x)
        end

        # Interaction with next point in y direction
        if j < n 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j + 1, k])
            push!(S, - α_y)

            push!(col_inds, lin_inds[i, j + 1, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, - α_y)

            # Self interaction 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, α_y)

            # Self interaction 
            push!(col_inds, lin_inds[i, j + 1, k])
            push!(row_inds, lin_inds[i, j + 1, k])
            push!(S, α_y)
        end
    
        # Interaction with next point in z direction
        if k < n 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k + 1])
            push!(S, - α_z)

            push!(col_inds, lin_inds[i, j, k + 1])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, - α_z)

            # Self interaction 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, α_z)

            # Self interaction 
            push!(col_inds, lin_inds[i, j, k + 1])
            push!(row_inds, lin_inds[i, j, k + 1])
            push!(S, α_z)
        end
    end

    # Assembling the sparse matrix
    A = sparse(row_inds, col_inds, S)
    # Forming the full operator by symmetrization.
    # Returning the problem
    return ReconstructionProblem(domains, Euclidean(), FactorizationOracle(cholesky(A)))
end 

# q is the number of levels of the subdivision
# α prescribes the conductivity coefficients by prescribing the edge-weights.
# It is a function such that α((x + y) / 2) is the conductivity between nodes in position x and y
# α is only called once on each location, meaning that it can be a random function
# β is the zeroth order term of the system 
# β(x) returns the zero order term in the location x.
function uniform3d_periodic_fd_poisson(q, α = (x, y, z) -> 1.0, β = (x, y, z) -> 1.0)
    periodic_α(x, y, z) = α(div(x, 1), div(y, 1), div(z, 1))
    periodic_β(x, y, z) = β(div(x, 1), div(y, 1), div(z, 1))
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
    row_inds = Int[]
    col_inds = Int[]
    S = Float64[]
    # contructing the vector that will store the domains
    domains = Vector{Domain{SVector{3, Float64}}}(undef, N)
    for i in 1 : n, j in 1 : n, k in 1 : n
        # Construct the Domain and add it to the list
        domains[lin_inds[i, j, k]] = 
            Domain(SVector((x[i], y[j], z[k])), 
                   lin_inds[i, j, k], 
                   Δx * Δy * Δz) 
        # adding self-interaction 2
        α_x = periodic_α(x[i] + Δx / 2, y[j], z[k]) / Δx^2
        α_y = periodic_α(x[i], y[j] + Δy / 2, z[k]) / Δy^2
        α_z = periodic_α(x[i], y[j], z[k] + Δz / 2) / Δz^2
        β_value = periodic_β(x[i], y[j], z[k])

        # Self interaction 
        push!(col_inds, lin_inds[i, j, k])
        push!(row_inds, lin_inds[i, j, k])
        push!(S, β_value)

        # Interaction with next point in x direction
        if i < n 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i + 1, j, k])
            push!(S, - α_x)

            push!(col_inds, lin_inds[i + 1, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, - α_x)

            # Self interaction 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, α_x)

            # Self interaction 
            push!(col_inds, lin_inds[i + 1, j, k])
            push!(row_inds, lin_inds[i + 1, j, k])
            push!(S, α_x)
        else
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[1, j, k])
            push!(S, - α_x)

            push!(col_inds, lin_inds[1, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, - α_x)

            # Self interaction 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, α_x)

            # Self interaction 
            push!(col_inds, lin_inds[1, j, k])
            push!(row_inds, lin_inds[1, j, k])
            push!(S, α_x)
        end

        # Interaction with next point in y direction
        if j < n 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j + 1, k])
            push!(S, - α_y)

            push!(col_inds, lin_inds[i, j + 1, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, - α_y)

            # Self interaction 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, α_y)

            # Self interaction 
            push!(col_inds, lin_inds[i, j + 1, k])
            push!(row_inds, lin_inds[i, j + 1, k])
            push!(S, α_y)
        else
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, 1, k])
            push!(S, - α_y)

            push!(col_inds, lin_inds[i, 1, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, - α_y)

            # Self interaction 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, α_y)

            # Self interaction 
            push!(col_inds, lin_inds[i, 1, k])
            push!(row_inds, lin_inds[i, 1, k])
            push!(S, α_y)
        end
        
        # Interaction with next point in z direction
        if k < n 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k + 1])
            push!(S, - α_z)

            push!(col_inds, lin_inds[i, j, k + 1])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, - α_z)

            # Self interaction 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, α_z)

            # Self interaction 
            push!(col_inds, lin_inds[i, j, k + 1])
            push!(row_inds, lin_inds[i, j, k + 1])
            push!(S, α_z)
        else
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, 1])
            push!(S, - α_z)

            push!(col_inds, lin_inds[i, j, 1])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, - α_z)

            # Self interaction 
            push!(col_inds, lin_inds[i, j, k])
            push!(row_inds, lin_inds[i, j, k])
            push!(S, α_z)

            # Self interaction 
            push!(col_inds, lin_inds[i, j, 1])
            push!(row_inds, lin_inds[i, j, 1])
            push!(S, α_z)
        end
    end

    # Assembling the sparse matrix
    A = sparse(row_inds, col_inds, S)
    # Returning the problem
    return ReconstructionProblem(domains, PeriodicEuclidean((1.0, 1.0, 1.0)), FactorizationOracle(cholesky(A)))
end 