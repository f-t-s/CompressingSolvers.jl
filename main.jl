using Random
Random.seed!(172)
using LinearAlgebra: Matrix
using CompressingSolvers
using SparseArrays
using LinearAlgebra
using GLMakie

# Setting up the test domains
ρ = 5
# ρ = Inf

q = 7
pb = uniform2d_dirichlet_fd_poisson(q)
# pb = uniform3d_dirichlet_fd_poisson(q)

# pb = uniform2d_neumann_fd_poisson(q)

# pb = uniform2d_fractional(q, 0.50, 1.0)
# pb = uniform2d_periodic_fd_poisson(q)
# pb = uniform2d_dirichlet_fd_poisson(q)

# path = "./gridap_models/demo.json"
# path = "./gridap_models/demo_refined.msh"
# # path = "./gridap_models/model.json"
# pb = gridap_poisson(path)
# pb, model, A = gridap_elasticity(path)
# 
# # 
rk, log = reconstruct(pb, ρ) 
# 
CompressingSolvers.compute_relative_error(rk, pb)

# reffe = ReferenceFE(lagrangian,Float64,order)
# # V = TestFESpace(model,reffe,dirichlet_tags=["boundary1","boundary2"])
# V = TestFESpace(model,reffe,dirichlet_tags=["sides", "bottom", "top", "square", "triangle", "circle", "square_c", "triangle_c", "circle_c", "sides_c"])
# # U = TrialFESpace(V,[0, 1])
# U = TrialFESpace(V,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) 
# Ω = Triangulation(model)
# dΩ = Measure(Ω,2*order)
# a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
# l(v) = 0
# op = AffineFEOperator(a,l,U,V)
# A = get_matrix(op)

# model = GmshDiscreteModel(path)
# # model = DiscreteModelFromFile(path)
# order = 1
# reffe = ReferenceFE(lagrangian,Float64,order)
# V = TestFESpace(model,reffe,dirichlet_tags=["boundary1", "boundary2"])
# # V = TestFESpace(model,reffe,dirichlet_tags=["volume"])
# # V = TestFESpace(model,reffe,dirichlet_tags=["sides",  "top", "square", "triangle", "circle"])
# U = TrialFESpace(V,[0, 1])
# # U = TrialFESpace(V, Float64[]) 
# Ω = Triangulation(model)
# dΩ = Measure(Ω,2*order)
# a(u,v) = ∫( ∇(v)⋅∇(u))dΩ
# l(v) = 0
# op = AffineFEOperator(a,l,U,V)
# A = get_matrix(op)
# # obtaining the physical locations of the dofs
# @assert get_free_dof_ids(U) == get_free_dof_ids(V)
# @show N = length(get_free_dof_ids(U))
# # Obtaining the locations and diameters associated to the dofs
# dof_locations = get_grid(model).node_coordinates[get_free_dof_ids(U)]
# # transforming the locations into SVectors
# dof_locations = [SVector(v[1], v[2], v[3]) for v in dof_locations]
# # dof_diameters = get_dof_diameters(get_grid(model))[get_free_dof_ids(U)]
# dof_diameters = 1.0
# domains = Vector{Domain{SVector{3, Float64}}}(undef, N) 
# for k = 1 : N
#     domains[k] = Domain(dof_locations[k], k, dof_diameters[k])
# end