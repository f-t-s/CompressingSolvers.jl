using Gridap: ReferenceFE, Triangulation, TestFESpace, TrialFESpace, Measure, ∫, ∇, ⋅, ε, tr, ⊙, AffineFEOperator, get_matrix, lagrangian, get_cell_coordinates, get_grid, get_free_dof_ids, DiscreteModelFromFile, VectorValue 
using Gridap.Io: to_json_file
using GridapGmsh: GmshDiscreteModel
using LinearAlgebra: norm, det, lu!, I
using StaticArrays: SVector
# model = GmshDiscreteModel("./gridap_models/demo_refined2.msh")
# order = 1
# reffe = ReferenceFE(lagrangian,Float64,order)
# V = TestFESpace(model,reffe,dirichlet_tags=["boundary1","boundary2"])
# U = TrialFESpace(V,[0,1])
# Ω = Triangulation(model)
# dΩ = Measure(Ω,2*order)
# a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
# l(v) = 0
# op = AffineFEOperator(a,l,U,V)
# uh = solve(op)
# writevtk(Ω,"demo",cellfields=["uh"=>uh])

function gmsh_model_to_json(gmsh_path, json_path)
    model = GmshDiscreteModel(gmsh_path)
    to_json_file(model, json_path)
end

function get_dof_diameters(grid)
    out = zeros(length(grid.node_coordinates))
    cell_coordinate_vector = get_cell_coordinates(grid)
    # Somehow the interface is broken?
    # cell_id_vector = get_cell_node_ids(grid)
    cell_id_vector = grid.cell_node_ids
    for (cell_index, cell_coordinates) in enumerate(cell_coordinate_vector)
        # The vector containing the node ids of the cell
        cell_ids = cell_id_vector[cell_index]
        for (node1_index, node1_coordinates) in enumerate(cell_coordinates)
            for (node2_index, node2_coordinates) in enumerate(cell_coordinates)
                # Obtaining the ids of the nodes
                node1_id = cell_ids[node1_index]
                node2_id = cell_ids[node2_index]
                # Computing the distance between the nodes
                dist = norm(node1_coordinates - node2_coordinates) 
                # Updating the distances
                out[node1_id] = max(out[node1_id], dist)
                out[node2_id] = max(out[node2_id], dist)
            end
        end
    end
    return 2 * out
end

function get_dof_volumes(grid)
    out = zeros(length(grid.node_coordinates))
    cell_coordinate_vector = get_cell_coordinates(grid)
    # Somehow the interface is broken?
    # cell_id_vector = get_cell_node_ids(grid)
    cell_id_vector = grid.cell_node_ids
    # |V| = 1 / 3! * |det(vcat([v1, v2, v3, v3], ones(1,4)))|
    mat = zeros(4, 4)
    for (cell_index, cell_coordinates) in enumerate(cell_coordinate_vector)
        # The vector containing the node ids of the cell
        cell_ids = cell_id_vector[cell_index]
        # preparing the determinant matrix
        @assert length(cell_coordinates) == 4
        mat[4, :] .= 1.0 
        for j = 1 : 4
            for i = 1 : 3
                mat[i, j] = cell_coordinates[j][i]
            end
        end
        vol = abs(det(lu!(mat))) / factorial(3)
        for node_id in cell_ids
            out[node_id] += vol
        end
    end
    return out
end

# A poisson problem optained from gridap
function gridap_poisson(path) 
    # model = GmshDiscreteModel(path)
    model = DiscreteModelFromFile(path)
    order = 1
    reffe = ReferenceFE(lagrangian,Float64,order)
    # The names and number of domains with dirichlet 
    # BC depends on the model used. TBD
    V = TestFESpace(model,reffe,dirichlet_tags=["boundary1", "boundary2"])
    U = TrialFESpace(V,[0, 1])
    Ω = Triangulation(model)
    dΩ = Measure(Ω,2*order)
    a(u,v) = ∫( ∇(v)⋅∇(u))dΩ
    l(v) = 0
    op = AffineFEOperator(a,l,U,V)
    A = get_matrix(op)
    # obtaining the physical locations of the dofs
    N = length(V.metadata.free_dof_to_node)
    # Obtaining the locations associated to dofs
    dof_locations = get_grid(model).node_coordinates[V.metadata.free_dof_to_node]
    # transforming the locations into SVectors
    dof_locations = [SVector(v[1], v[2], v[3]) for v in dof_locations]
    # Obtaining the volumes associated to dofs
    dof_volumes = get_dof_volumes(get_grid(model))[V.metadata.free_dof_to_node]
    domains = Vector{Domain{SVector{3, Float64}}}(undef, N) 
    for k = 1 : N
        domains[k] = Domain(dof_locations[k], k, dof_volumes[k])
    end

    # Returning the reconstruction problem
    return ReconstructionProblem(domains, Euclidean(), FactorizationOracle(cholesky(A)))
end

# A linear elasticity problem optained from gridap
# For now, this does not interfact with reconstruction, since # reconstruction is not able to handle multi-output functions # at this point
function gridap_elasticity(path) 
    model = GmshDiscreteModel(path)
    # model = DiscreteModelFromFile(path)
    order = 1

    reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
    V0 = TestFESpace(model,reffe;
        conformity=:H1,
        dirichlet_tags=["boundary1","boundary2"],
        dirichlet_masks=[(true,false,false), (true,true,true)])

    g1(x) = VectorValue(0.005,0.0,0.0)
    g2(x) = VectorValue(0.0,0.0,0.0)

    U = TrialFESpace(V0, [g1, g2])

    # constitutive law
    # values used to be const in original code
    E = 70.0e9
    ν = 0.33
    λ = (E*ν)/((1+ν)*(1-2*ν))
    μ = E/(2*(1+ν))
    σ(ε) = λ*tr(ε)*one(ε) + 2*μ*ε

    degree = 2*order
    Ω = Triangulation(model)
    dΩ = Measure(Ω,degree)

    # Defining the bilinear form
    # a(u,v) = ∫( ε(v) ⊙ (σ∘ε(u)) )*dΩ
    a(u,v) = ∫( ε(v) ⊙ (σ∘ε(u)) )*dΩ
    l(v) = 0

    op = AffineFEOperator(a,l,U,V0)
    A = get_matrix(op)
    # obtaining the physical locations of the dofs
    N = length(V0.metadata.free_dof_to_node)
    # Obtaining the locations associated to dofs
    dof_locations = get_grid(model).node_coordinates[V0.metadata.free_dof_to_node]
    # transforming the locations into SVectors
    dof_locations = [SVector(v[1], v[2], v[3]) for v in dof_locations]
    # Obtaining the volumes associated to dofs
    dof_volumes = get_dof_volumes(get_grid(model))[V0.metadata.free_dof_to_node]
    domains = Vector{Domain{SVector{3, Float64}}}(undef, N) 
    for k = 1 : N
        domains[k] = Domain(dof_locations[k], k, dof_volumes[k])
    end

    # Returning the reconstruction problem
    # The matrix should be symmetric and seems to be to very high order
    @assert norm(A - A') / norm(A) ≤ 1e-14
    #A = (A + A') / 2

    return ReconstructionProblem(domains, Euclidean(), FactorizationOracle(cholesky(A))), model, A
end