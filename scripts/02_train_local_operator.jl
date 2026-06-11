using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using TrojanBiLO
using Lux
using Random
using Optimisers
using Zygote
using Statistics
using ProgressMeter
using JLD2
using Printf
using GLMakie

Makie.inline!(true)

const MU = MU_SUN_JUPITER
L4 = l4_position(MU)
center = vcat(collect(L4), [0.0, 0.0, 0.0])
radius = [0.01, 0.01, 0.0, 0.005, 0.005, 0.0]
tr = TrustRegion(center, radius)

t_grid = collect(range(0.0, 6.0; length=32))

rng = MersenneTwister(1)
op = build_operator(center; p_dim=32, x_scale=radius, t_scale=6.0)
ps, st = init_params(op, rng)

const N_STEPS = 800
const N_INNER = 16
const LR = 3e-3

val_rng = MersenneTwister(99)
val_batch = sample_x0(tr, val_rng, 8)
val_loss(p) = supervised_loss(op, p, st, val_batch, t_grid; mu=MU, λ_J=0.0)

opt_state = Optimisers.setup(Optimisers.Adam(LR), ps)
loss_hist = Float64[]
data_hist = Float64[]
jac_hist = Float64[]

function loss_breakdown(op, ps, st, batch, t_grid; mu, λ_J)
    N, T = size(batch, 2), length(t_grid)
    Ld = 0.0; Lj = 0.0
    for k in 1:N
        x0 = batch[:, k]
        X = propagate_diff(x0, t_grid; mu=mu)
        Y, _ = forward_grid(op, ps, st, x0, t_grid)
        Ld += sum(abs2, Y .- X) / T
        J0 = jacobi_constant(x0, mu)
        jp = 0.0
        for j in 1:T; jp += (jacobi_constant(Y[:, j], mu) - J0)^2; end
        Lj += jp / T
    end
    return Ld / N, λ_J * Lj / N
end

let
    nprobe = 32
    pr_batch = sample_x0(tr, MersenneTwister(7), nprobe)
    spread = 0.0
    peak = 0.0
    for k in 1:nprobe
        X = propagate_diff(pr_batch[:, k], t_grid; mu=MU)
        spread += sum(abs2, X .- center) / length(t_grid)
        peak = max(peak, maximum(abs, X .- center))
    end
    @printf "true trajectory mean-square deviation from x_ref = %.3e peak |Delta| = %.3e\n" (spread / nprobe) peak

    Ld0, Lj0 = loss_breakdown(op, ps, st, val_batch, t_grid; mu=MU, λ_J=0.0)
    @printf "Initial Value:  data = %.3e   lambda_J·jac = %.3e   total = %.3e\n" Ld0 Lj0 (Ld0 + Lj0)
end

@info "training local surrogate over fixed trust region"
function train!(op, ps, st, tr, rng, t_grid, opt_state,
                loss_hist, data_hist, jac_hist,
                n_steps, n_inner, val_batch)
    @showprogress for k in 1:n_steps
        x0_batch = sample_x0(tr, rng, n_inner)
        _, gs = Zygote.withgradient(
            p -> supervised_loss(op, p, st, x0_batch, t_grid; mu=MU, λ_J=0.0), ps
        )
        opt_state, ps = Optimisers.update!(opt_state, ps, gs[1])
        if k % 10 == 0
            Ld, Lj = loss_breakdown(op, ps, st, val_batch, t_grid; mu=MU, λ_J=0.0)
            push!(loss_hist, Ld + Lj)
            push!(data_hist, Ld)
            push!(jac_hist,  Lj)
            @printf "  step %4d   data = %.3e   lambda_J·jac = %.3e   total = %.3e\n" k Ld Lj (Ld + Lj)
        end
    end
    return ps, st, opt_state
end
ps, st, opt_state = train!(op, ps, st, tr, rng, t_grid, opt_state,
                           loss_hist, data_hist, jac_hist,
                           N_STEPS, N_INNER, val_batch)

x0_test = center .+ 0.5 .* radius
X_true = propagate_diff(x0_test, t_grid; mu=MU)
X_pred, _ = forward_grid(op, ps, st, x0_test, t_grid)

err = sqrt(mean(abs2, X_pred .- X_true))
@printf "\n Hold-out trajectory RMS error = %.3e \n" err

fig = Figure(size=(1300, 400))
ax1 = Axis(fig[1, 1], title="inner loss (val)", yscale=log10,
           xlabel="step", ylabel="loss")
xs = 10 .* (1:length(loss_hist))
lines!(ax1, xs, data_hist .+ eps(); color=:steelblue, label="data")
lines!(ax1, xs, jac_hist .+ eps(); color=:tomato, label="λ_J·jac")
lines!(ax1, xs, loss_hist .+ eps(); color=:black, linestyle=:dash, label="total")
axislegend(ax1; position=:rt)

ax2 = Axis(fig[1, 2], title="Surrogate vs Vern9",
           xlabel="x", ylabel="y")
lines!(ax2, X_true[1, :], X_true[2, :]; color=:black, label="Vern9")
lines!(ax2, X_pred[1, :], X_pred[2, :]; color=:steelblue, linestyle=:dash, label="Surrogate")
scatter!(ax2, [L4[1]], [L4[2]]; color=:red, markersize=10, label="L4")
axislegend(ax2; position=:rb)
ax2.aspect = DataAspect()

ax3 = Axis(fig[1, 3], title="Error vs Time ",
           xlabel="t", ylabel="|pred − true|", yscale=log10)
err_t = vec(sqrt.(sum(abs2, X_pred .- X_true; dims=1)))
lines!(ax3, t_grid, err_t .+ eps(); color=:steelblue)

mkpath(joinpath(@__DIR__, "..", "figures"))
save(joinpath(@__DIR__, "..", "figures", "02_surrogate_sanity.png"), fig)

mkpath(joinpath(@__DIR__, "..", "data"))
outfile = joinpath(@__DIR__, "..", "data", "local_operator.jld2")
@save outfile ps st center radius t_grid loss_hist
@info "trained local surrogate" outfile
