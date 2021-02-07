import SparseArrays.SparseMatrixCSC
import SparseArrays: findnz, dropzeros!, spdiagm
import StaticArrays: SVector

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
    ticks = ρh : (2 * ρh) : (1 - ρh)
    # if ticks is empty, add a single entry to it, resulting in all nodes being summarized in the same supernode
    if isempty(ticks) 
        ticks = [zero(ρh)]
    end
    return SVector{2}.([[x; y] for x in ticks for y in ticks])
end

function supernodal_aggregation_square(domains, scales, basis_functions, ρ)
    ##################################################################
    # Constructing supernodes 
    ##################################################################
    # The supernodes corresponding to different columns of 𝐋
    basis_supernodes = construct_supernodes.(aggregation_centers_square.(ρ * scales), basis_functions)
    # Supernodes corresponding to different rows of 𝐋
    domain_supernodes = construct_supernodes(aggregation_centers_square(ρ * scales[end]), domains)
    # Multicolor ordering 
    multicolor_ordering = construct_multicolor_ordering(basis_supernodes, 1.5 * ρ * scales)
    return basis_supernodes, domain_supernodes, multicolor_ordering
end



# Create a finite difference Laplacian problem on a quadratic mesh using sudivision, with dirichlet boundary conditions.
# q: total number of subdivisions, leading to a number dofs given by 2^{qd}
# α: the coefficient function. an edge between x and y will have conductivity (α(x) + α(y)) / 2
# Possibly remore the implicit choice of ρ
function FD_Laplacian_subdivision_2d(q, ρ = 2.0, α = x -> 1)
    n = 2 ^ q
    N = n ^ 2  
    Δx = Δy = 1 / (n + 1)
    ##################################################################
    # Construct the Laplace operator
    ##################################################################
    # Create the sparsity pattern of the Laplace operator 
    A = spdiagm(-n => -ones(N-n), -1 => -ones(N-1) .* (mod.(1 : (N - 1), n) .!= 0), 0 => 4 * ones(N), 1 => -ones(N-1) .* (mod.(1 : (N - 1), n) .!= 0), n => -ones(N-n)) / Δx^2
    dropzeros!(A)
    A .*= 0

    # fill the Laplace operator with nonzeros according o the sparsity pattern
    for (i, j, val) in zip(findnz(A)...)
        # only look at edges
        if i != j
            val = (α(center(domains[i])) + α(center(domains[j]))) / 2
            A[i, i] += val   
            A[j, j] += val
            A[i, j] -= val
            A[j, i] -= val
        end
    end

    ##################################################################
    # Construct the domain decomposition
    ##################################################################
    domains, scales, basis_functions = subdivision_2d(q)

    ##################################################################
    # Constructing supernodes 
    ##################################################################
    # The supernodes corresponding to different columns of 𝐋
    basis_supernodes = construct_supernodes.(aggregation_centers_square.(ρ * scales), basis_functions)
    # Supernodes corresponding to different rows of 𝐋
    domain_supernodes = construct_supernodes(aggregation_centers_square(ρ * scales[end]), domains)
    # Multicolor ordering 
    multicolor_ordering = construct_multicolor_ordering(basis_supernodes, 1.5 * ρ * scales)
    return A, domains, scales, basis_functions, basis_supernodes, domain_supernodes, multicolor_ordering
end