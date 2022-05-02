import SparseArrays.SparseMatrixCSC
import SparseArrays: findnz, dropzeros!, spdiagm
import StaticArrays: SVector
using NearestNeighbors: BallTree, KDTree
using Distances: PeriodicEuclidean

# q: total number of subdivisions, leading to a number dofs given by 2^{qd}
function subdivision_2d(q)
    ##################################################################
    # Construct the domains
    ##################################################################
    n = 2 ^ q
    N = n ^ 2  
    Δx = Δy = 1 / (n + 1)
    # The evaluation points of the finite difference discretization.
    domains = array2domains(mapreduce(identity, hcat, [[x; y] for x in Δx .* (1/2 : (n - 1/2)) for y in Δy .* (1/2 : (n - 1/2))]))

    ##################################################################
    # Construct the domain decomposition
    ##################################################################

    # Note that the ids of all bu the finest scale domains are arbitrary
    next_id = maximum(id.(domains)) + 1
    for k = (q - 1) : -1 : 1
        new_domains = Vector{eltype(domains)}(undef, 2 ^ (2 * k))
        # return assignment to a node on the coarser scale 
        # TODO: Still need to test that ij -> ji is doing the right thing
        function return_assignment(domain)
            i, j = Int.(div.(center(domain), 1 / 2^k) .+ 1)
            return LinearIndices(zeros(2^k, 2^k))[i, j]
        end
        # go through all domains on the next finer level 
        for domain in domains 
            parent_index = return_assignment(domain)
            # eithercreate the domain if it doesn't exist yet
            if !isassigned(new_domains, parent_index)
                new_domains[parent_index] = Domain([domain], next_id)
                next_id += 1
            # or else add the new child
            else
                new_domains[parent_index] = add_children(new_domains[parent_index], [domain])
            end
        end
        domains = new_domains
    end
    scales = 1 ./ (2 .^ (1 : q))

    ##################################################################
    # Construct the basis functions 
    ##################################################################
    basis_functions = compute_basis_functions(domains)

    return domains, scales, basis_functions 
end

# function for selecting the aggregation centers in the square 
function aggregation_centers_square(ρh) 
    if ρh < Inf
        ticks = ρh : (ρh) : (1 - ρh)
    else
        ticks = [zero(ρh)]
    end
    # if ticks is empty, add a single entry to it, resulting in all nodes being summarized in the same supernode
    if isempty(ticks) 
        ticks = [zero(ρh)]
    end
    return SVector{2}.([[x; y] for x in ticks for y in ticks])
end

# Create a finite difference Laplacian problem on a quadratic mesh using sudivision, with dirichlet boundary conditions.
# q: total number of subdivisions, leading to a number dofs given by 2^{qd}
# α: the coefficient function. an edge between x and y will have conductivity (α(x) + α(y)) / 2
# Possibly remore the implicit choice of ρ
function FD_Laplacian_subdivision_2d(q, ρ = 2.0, α = x -> 1)
    n = 2 ^ q
    N = n ^ 2  
    Δx = Δy = 1 / (n + 1)

    # actual multiscale domains are provided by subdivision_2d. fine_domains only 
    # there fore the 

    ##################################################################
    # Construct the Laplace operator
    ##################################################################
    # Create the sparsity pattern of the Laplace operator 
    A = spdiagm(-n => -ones(N-n), -1 => -ones(N-1) .* (mod.(1 : (N - 1), n) .!= 0), 0 => 4 * ones(N), 1 => -ones(N-1) .* (mod.(1 : (N - 1), n) .!= 0), n => -ones(N-n)) # / Δx^2

    # Important to multiply A.nzval and not A, to prevent dropping of structural nonzeros.
    A.nzval .*= 0
   ##################################################################
    # Construct the domain decomposition
    ##################################################################
    domains, scales, basis_functions = subdivision_2d(q)

    # collect the finest Domains and sorts them to be in accordance with the construction of the matrix dofs
    fine_domains = sort(gather_descendants(domains), by=id)

    for (i, j, val) in zip(findnz(A)...)
        # only look at edges
        if i != j
            val = (α(center(fine_domains[i])) + α(center(fine_domains[j]))) / 2 / Δx^2
            A[i, i] += val   
            A[j, j] += val
            A[i, j] -= val
            A[j, i] -= val
        end
    end
    # # adding zero order term for now
    # for i = 1 : size(A, 1)
    #     A[i, i] += 1.0
    # end

    # constructing the tree function
    function tree_function(x)
        return KDTree(x)
    end


    ##################################################################
    # Constructing the multicolor ordering
    ##################################################################
    # The supernodes corresponding to different columns of 𝐋
    multicolor_ordering = construct_multicolor_ordering(basis_functions, ρ * scales, tree_function)

    return A, domains, scales, basis_functions, multicolor_ordering, fine_domains, tree_function
end

# Create a finite difference Laplacian problem on a quadratic mesh using sudivision, with dirichlet boundary conditions.
# q: total number of subdivisions, leading to a number dofs given by 2^{qd}
# α: the coefficient function. an edge between x and y will have conductivity (α(x) + α(y)) / 2
# Possibly remore the implicit choice of ρ
function FD_periodic_Laplacian_subdivision_2d(q, ρ = 2.0, α = x -> 1)
    n = 2 ^ q
    N = n ^ 2  
    Δx = Δy = 1 / (n + 1)
    h = Δx
    # actual multiscale domains are provided by subdivision_2d. fine_domains only 
    # there fore the 

    ##################################################################
    # Construct the Laplace operator
    ##################################################################
    # Create the sparsity pattern of the Laplace operator 
    lin_inds = LinearIndices((n, n))
    row_inds = Int[]
    col_inds = Int[]
    S = Float64[]
    for i in 1 : n, j in 1 : n
        # adding self-interaction 2
        push!(col_inds, lin_inds[mod(i - 1, n) + 1, mod(j - 1, n) + 1]) 
        push!(row_inds, lin_inds[mod(i - 1, n) + 1, mod(j - 1, n) + 1]) 
        push!(S, 4.0) 
        # interaction to next element in i direction
        push!(col_inds, lin_inds[mod(i - 1, n) + 1, mod(j - 1, n) + 1]) 
        push!(row_inds, lin_inds[mod(i - 1 + 1, n) + 1, mod(j - 1, n) + 1]) 
        push!(S, -1.0)

        # interaction to previous element in i direction
        push!(col_inds, lin_inds[mod(i - 1, n) + 1, mod(j - 1, n) + 1])
        push!(row_inds, lin_inds[mod(i - 1 - 1, n) + 1, mod(j - 1, n) + 1]) 
        push!(S, -1.0)

        # interaction to next element in j direction
        push!(col_inds, lin_inds[mod(i - 1, n) + 1, mod(j - 1, n) + 1]) 
        push!(row_inds, lin_inds[mod(i - 1, n) + 1, mod(j - 1 + 1, n) + 1]) 
        push!(S, -1.0)

        # interaction to previous element in j direction
        push!(col_inds, lin_inds[mod(i - 1, n) + 1, mod(j - 1, n) + 1]) 
        push!(row_inds, lin_inds[mod(i - 1, n) + 1, mod(j - 1 - 1, n) + 1]) 
        push!(S, -1.0)
    end
    A = sparse(row_inds, col_inds, S) / (h^2)

    # Important to multiply A.nzval and not A, to prevent dropping of structural nonzeros.
    A.nzval .*= 0
   ##################################################################
    # Construct the domain decomposition
    ##################################################################
    domains, scales, basis_functions = subdivision_2d(q)

    # collect the finest Domains and sorts them to be in accordance with the construction of the matrix dofs
    fine_domains = sort(gather_descendants(domains), by=id)

    for (i, j, val) in zip(findnz(A)...)
        # only look at edges
        if i < j
            val = (α(center(fine_domains[i])) + α(center(fine_domains[j]))) / 2 / Δx^2
            A[i, i] += val   
            A[j, j] += val
            A[i, j] -= val
            A[j, i] -= val
        end
    end

    # adding zero order term for now
    for i = 1 : size(A, 1)
        A[i, i] += 1.0 + rand()
    end

    # constructing the tree function
    function tree_function(x)
        return BallTree(x, PeriodicEuclidean([1.0, 1.0]))
    end


    ##################################################################
    # Constructing the multicolor ordering
    ##################################################################
    # The supernodes corresponding to different columns of 𝐋
    multicolor_ordering = construct_multicolor_ordering(basis_functions, ρ * scales, tree_function)

    return A, domains, scales, basis_functions, multicolor_ordering, fine_domains, tree_function
end