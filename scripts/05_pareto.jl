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
t_grid = collect(range(0.0, 6.0; length=32))

const N_OUTER = 200
const N_CYCLES = 10
const N_INNER = 4
const TARGET = :tadpole
const T_VALIDATE = 200.0

bilo_settings = [
    (label = "BiLO n_refit=2",  n_refit = 2),
    (label = "BiLO n_refit=8",  n_refit = 8),
    (label = "BiLO n_refit=30", n_refit = 30),
    (label = "BiLO n_refit=80", n_refit = 80),
]

results = NamedTuple[]

@info "JIT warmup"
let
    rng = MersenneTwister(0)
    op = build_operator(x0_init; p_dim=32, x_scale=radius, t_scale=6.0)
    ps, st = init_params(op, rng)
    tr = TrustRegion(copy(x0_init), radius)
    bilo_run(op, ps, st, x0_init, tr, t_grid;
             rng=rng, n_outer=4, n_cycles=2, n_refit=1, n_inner=2,
             lr_θ=3e-3, lr_x0=5e-3, lambda_J=0.0,
             target=TARGET, mu=MU, verbose=false)
    direct_search(x0_init, t_grid;
                  n_steps=4, lr=5e-3, target=TARGET, mu=MU, verbose=false)
end

function run_bilo_point(setting, x0_init, radius, t_grid)
    rng = MersenneTwister(2)
    op = build_operator(x0_init; p_dim=32, x_scale=radius, t_scale=6.0)
    ps, st = init_params(op, rng)
    tr = TrustRegion(copy(x0_init), radius)

    t0 = time()
    ps, st, x0, hist, _ = bilo_run(
        op, ps, st, x0_init, tr, t_grid;
        rng=rng, n_outer=N_OUTER, n_cycles=N_CYCLES,
        n_refit=setting.n_refit, n_inner=N_INNER,
        lr_θ=3e-3, lr_x0=5e-3, lambda_J=0.0,
        target=TARGET, mu=MU, verbose=false,
    )
    return (wall = time() - t0, x0 = x0, hist = hist)
end

for s in bilo_settings
    @info "Running $(s.label)"
    r = run_bilo_point(s, x0_init, radius, t_grid)

    sol = propagate_cr3bp(r.x0, (0.0, T_VALIDATE); mu=MU, saveat=0.1)
    cls = classify_trajectory(sol; mu=MU)
    amp = rad2deg(libration_amplitude(sol; mu=MU))
    fLo = r.hist[end].outer

    @printf "  %-18s  total = %6.2f s  L_outer = %.3e  amp = %5.1f°  class = %s\n" s.label r.wall fLo amp cls
    push!(results, (; label=s.label, n_refit=s.n_refit,
                     wall=r.wall, L_outer=fLo, amp=amp, class=cls, x0=copy(r.x0)))
end


@info "Running direct baseline"
t0 = time()
x0_d, hist_d = direct_search(
    x0_init, t_grid;
    n_steps=N_OUTER, lr=5e-3, target=TARGET, mu=MU, verbose=false,
)
wall_d = time() - t0
sol_d = propagate_cr3bp(x0_d, (0.0, T_VALIDATE); mu=MU, saveat=0.1)
cls_d = classify_trajectory(sol_d; mu=MU)
amp_d = rad2deg(libration_amplitude(sol_d; mu=MU))
fLo_d = hist_d[end].outer
@printf "  %-18s  wall = %6.2f s  L_outer = %.3e  amp = %5.1f°  class = %s\n" "direct" wall_d fLo_d amp_d cls_d
push!(results, (; label="direct", n_refit=0,
                 wall=wall_d, L_outer=fLo_d, amp=amp_d, class=cls_d, x0=copy(x0_d)))

println("\n Pareto summary ")
@printf "%-18s  %8s  %12s  %8s  %s\n" "method" "wall(s)" "L_outer" "amp(°)" "class"
for r in results
    @printf "%-18s  %8.2f  %12.3e  %8.2f  %s\n" r.label r.wall r.L_outer r.amp r.class
end

fig = Figure(size=(900, 400))

ax1 = Axis(fig[1, 1]; title="speed / precision Pareto",
           xlabel="wall time  (s, log)", ylabel="final outer loss  (log)",
           xscale=log10, yscale=log10)
ax2 = Axis(fig[1, 2]; title="speed / amplitude",
           xlabel="wall time  (s, log)", ylabel="long-horizon amplitude  (°)",
           xscale=log10)

bilo_results = filter(r -> startswith(r.label, "BiLO"), results)
direct_results = filter(r -> r.label == "direct", results)
sort!(bilo_results; by = r -> r.wall)

bilo_wall = [r.wall for r in bilo_results]
bilo_lout = [r.L_outer for r in bilo_results]
bilo_amp = [r.amp for r in bilo_results]
dir_wall = [r.wall for r in direct_results]
dir_lout = [r.L_outer for r in direct_results]
dir_amp = [r.amp for r in direct_results]

scatter!(ax1, bilo_wall, bilo_lout; color=:steelblue, markersize=14, label="BiLO sweep")
lines!(ax1, bilo_wall, bilo_lout; color=:steelblue, linestyle=:dash)
scatter!(ax1, dir_wall,  dir_lout;  color=:tomato,    markersize=14, label="direct")
for r in bilo_results
    text!(ax1, r.wall, r.L_outer;
          text="  n=$(r.n_refit)",
          align=(:left, :bottom), fontsize=10)
end
for r in direct_results
    text!(ax1, r.wall, r.L_outer;
          text="direct  ",
          align=(:right, :bottom), fontsize=10)
end
xlims!(ax1, 0.05, 200)
axislegend(ax1; position=:rt)

scatter!(ax2, bilo_wall, bilo_amp; color=:steelblue, markersize=14, label="BiLO sweep")
lines!(ax2, bilo_wall, bilo_amp; color=:steelblue, linestyle=:dash)
scatter!(ax2, dir_wall,  dir_amp;  color=:tomato,    markersize=14, label="direct")
for r in bilo_results
    text!(ax2, r.wall, r.amp;
          text="  n=$(r.n_refit)",
          align=(:left, :bottom), fontsize=10)
end
for r in direct_results
    text!(ax2, r.wall, r.amp;
          text="direct  ",
          align=(:right, :bottom), fontsize=10)
end
hlines!(ax2, [rad2deg(2 * 0.05)]; color=:gray, linestyle=:dot, label="target ≈ 5.7°")
xlims!(ax2, 0.05, 200)
axislegend(ax2; position=:rt)

mkpath(joinpath(@__DIR__, "..", "figures"))
save(joinpath(@__DIR__, "..", "figures", "05_pareto.png"), fig)

mkpath(joinpath(@__DIR__, "..", "data"))
outfile = joinpath(@__DIR__, "..", "data", "pareto.jld2")
@save outfile results
@info "Pareto sweep complete" outfile
