# ============================================================================
# Tests: equilibrium at L4, Jacobi conservation under Vern9,
# differentiability of `propagate_diff`, and classifier behavior on a
# small perturbation. Run with:
#
#   julia --project=. -e 'using Pkg; Pkg.test()'
# ============================================================================

using TrojanBiLO
using Test, LinearAlgebra, Zygote, Random

@testset "L4 is an equilibrium" begin
    L4    = l4_position()
    state = vcat(collect(L4), [0.0, 0.0, 0.0])
    @test norm(cr3bp_rhs(state, MU_SUN_JUPITER, 0.0)) < 1e-12
end

@testset "Jacobi constant conserved under Vern9" begin
    L4  = l4_position()
    x0  = vcat(collect(L4), [0.0, 0.0, 0.0]) .+ 0.005
    sol = propagate_cr3bp(x0, (0.0, 10.0); saveat=0.1)
    J   = [jacobi_constant(u) for u in sol.u]
    @test maximum(abs, J .- J[1]) < 1e-8
end

@testset "propagate_diff is differentiable in x0" begin
    L4     = l4_position()
    x0     = collect(vcat(collect(L4), [0.0, 0.0, 0.0]) .+ 0.005)
    t_grid = collect(range(0.0, 2.0; length=8))
    g = Zygote.gradient(x -> sum(propagate_diff(x, t_grid)), x0)[1]
    @test g !== nothing
    @test all(isfinite, g)
    @test norm(g) > 0
end

@testset "Surrogate forward shapes" begin
    L4     = l4_position()
    center = collect(vcat(collect(L4), [0.0, 0.0, 0.0]))
    op     = build_operator(center; p_dim=8)
    ps, st = init_params(op, MersenneTwister(0))
    t_grid = collect(range(0.0, 1.0; length=4))
    Y, _   = forward_grid(op, ps, st, center, t_grid)
    @test size(Y) == (6, length(t_grid))
end

@testset "Classifier on small L4 perturbation" begin
    L4  = l4_position()
    x0  = collect(vcat(collect(L4), [0.0, 0.0, 0.0])) .+ [0.002, 0.0, 0.0, 0.0, 0.0, 0.0]
    sol = propagate_cr3bp(x0, (0.0, 30.0); saveat=0.05)
    @test classify_trajectory(sol) in (TADPOLE, BOUND)
end
