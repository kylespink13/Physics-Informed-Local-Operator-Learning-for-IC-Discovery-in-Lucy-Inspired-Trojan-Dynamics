using GLMakie

function plot_trajectory!(ax, sol; mu=MU_SUN_JUPITER,
                          label="", color=:steelblue)
    xs = [u[1] for u in sol.u]
    ys = [u[2] for u in sol.u]
    lines!(ax, xs, ys; color=color, label=label)

    scatter!(ax, [-mu], [0.0]; color=:orange, markersize=14, label="Sun")
    scatter!(ax, [1 - mu], [0.0]; color=:brown,  markersize=10, label="Jupiter")

    L4 = l4_position(mu); L5 = l5_position(mu)
    scatter!(ax, [L4[1], L5[1]], [L4[2], L5[2]];
             color=:black, marker=:cross, markersize=12, label="L4/L5")

    ax.aspect = DataAspect()
    ax.xlabel = "x"
    ax.ylabel = "y"
    return ax
end

function plot_loss_curve!(ax, history; field::Symbol=:outer,
                          label="", color=:steelblue)
    isempty(history) && return ax
    haskey(history[1], field) || return ax
    steps  = [h.step for h in history]
    values = [getproperty(h, field) for h in history]
    lines!(ax, steps, values; color=color, label=label)
    ax.yscale = log10
    ax.xlabel = "outer step"
    ax.ylabel = String(field) * " loss"
    return ax
end

function plot_jacobi_drift!(ax, sol; mu=MU_SUN_JUPITER,
                            label="", color=:steelblue)
    J0   = jacobi_constant(sol.u[1], mu)
    drift = [abs(jacobi_constant(u, mu) - J0) + eps() for u in sol.u]
    lines!(ax, sol.t, drift; color=color, label=label)
    ax.yscale = log10
    ax.xlabel = "t  (nondim)"
    ax.ylabel = "|C - C_{0}|"
    return ax
end
