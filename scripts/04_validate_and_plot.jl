using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using TrojanBiLO
using JLD2
using Printf
using GLMakie

Makie.inline!(true)

const MU = MU_SUN_JUPITER

infile = joinpath(@__DIR__, "..", "data", "bilo_run.jld2")
@load infile x0_bilo x0_direct

candidates = Dict("BiLO" => x0_bilo, "direct" => x0_direct)
const T_VALIDATE = 200.0

results = Dict{String, Any}()
for (name, x0) in candidates
    sol = propagate_cr3bp(x0, (0.0, T_VALIDATE); mu=MU, saveat=0.1)
    label = classify_trajectory(sol; mu=MU)
    amp = rad2deg(libration_amplitude(sol; mu=MU))
    @printf "%-7s  class = %-10s  amplitude = %6.1f°\n" name label amp
    results[name] = (sol=sol, label=label, amp=amp)
end

L4 = l4_position(MU)
zoom = 0.25

fig1 = Figure(size=(900, 450))
ax_b = Axis(fig1[1, 1]; title="BiLO  (class=$(results["BiLO"].label))",
            xlabel="x", ylabel="y", aspect=DataAspect())
ax_d = Axis(fig1[1, 2]; title="direct  (class=$(results["direct"].label))",
            xlabel="x", ylabel="y", aspect=DataAspect())
for (ax, name, color) in [(ax_b, "BiLO", :steelblue), (ax_d, "direct", :tomato)]
    sol = results[name].sol
    xs = [u[1] for u in sol.u]
    ys = [u[2] for u in sol.u]
    lines!(ax, xs, ys; color=color, label=name)
    scatter!(ax, [L4[1]], [L4[2]]; color=:black, marker=:cross, markersize=12, label="L4")
    xlims!(ax, L4[1] - zoom, L4[1] + zoom)
    ylims!(ax, L4[2] - zoom, L4[2] + zoom)
end
axislegend(ax_b; position=:lt); axislegend(ax_d; position=:lt)

mkpath(joinpath(@__DIR__, "..", "figures"))
save(joinpath(@__DIR__, "..", "figures", "04_trajectory_comparison.png"), fig1)

fig2 = Figure(size=(700, 350))
ax_j = Axis(fig2[1, 1]; title="Jacobi-Constant Drift (Vern9 Validation)")
plot_jacobi_drift!(ax_j, results["BiLO"].sol;   mu=MU, label="BiLO",   color=:steelblue)
plot_jacobi_drift!(ax_j, results["direct"].sol; mu=MU, label="direct", color=:tomato)
axislegend(ax_j; position=:rt)
save(joinpath(@__DIR__, "..", "figures", "04_jacobi_drift.png"), fig2)

@info "Validation complete"
