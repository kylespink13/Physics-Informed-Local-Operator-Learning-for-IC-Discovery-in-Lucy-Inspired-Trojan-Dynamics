using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using TrojanBiLO
using Lux
using Random
using JLD2
using Printf
using GLMakie

Makie.inline!(true)

const MU = MU_SUN_JUPITER

L4 = l4_position(MU)
x0_init = vcat(collect(L4), [0.0, 0.0, 0.0])
radius = [0.01, 0.01, 0.0, 0.005, 0.005, 0.0]
tr = TrustRegion(copy(x0_init), radius)
t_grid = collect(range(0.0, 6.0; length=32))

const N_STEPS = 200
const TARGET = :tadpole

rng = MersenneTwister(2)
op = build_operator(x0_init; p_dim=32, x_scale=radius, t_scale=6.0)
ps, st = init_params(op, rng)

@info "Running solver-in-the-loop BiLO ($N_STEPS steps)"
t_bilo = @elapsed begin
    global ps, st, x0_bilo, hist_bilo, tr_final = bilo_run(
        op, ps, st, x0_init, tr, t_grid;
        rng=rng, n_outer=N_STEPS, n_cycles=10, n_refit=12, n_inner=4,
        lr_θ=3e-3, lr_x0=5e-3, λ_J=0.0,
        target=TARGET, mu=MU,
    )
end
@printf "BiLO final x0 = %s   total time = %.2f s   (%.3f s/step)\n" string(round.(x0_bilo; digits=5)) t_bilo (t_bilo / N_STEPS)

@info "running direct differentiable baseline ($N_STEPS steps)"
t_direct = @elapsed begin
    global x0_direct, hist_direct = direct_search(
        x0_init, t_grid;
        n_steps=N_STEPS, lr=5e-3, target=TARGET, mu=MU,
    )
end
@printf "direct final x0 = %s   total time = %.2f s   (%.3f s/step)\n" string(round.(x0_direct; digits=5)) t_direct (t_direct / N_STEPS)
@printf "\n BiLO speedup per outer step = %.2fx \n\n" (t_direct / t_bilo)

fig = Figure(size=(700, 400))
ax  = Axis(fig[1, 1]; title="outer loss vs step",
           xlabel="outer step", ylabel="outer loss",
           yscale=log10)
plot_loss_curve!(ax, hist_bilo;   field=:outer, label="BiLO",   color=:steelblue)
plot_loss_curve!(ax, hist_direct; field=:outer, label="direct", color=:tomato)
axislegend(ax; position=:rt)

mkpath(joinpath(@__DIR__, "..", "figures"))
save(joinpath(@__DIR__, "..", "figures", "03_loss_curves.png"), fig)

mkpath(joinpath(@__DIR__, "..", "data"))
outfile = joinpath(@__DIR__, "..", "data", "bilo_run.jld2")
@save outfile x0_bilo hist_bilo x0_direct hist_direct
@info "Bilevel search complete" outfile
