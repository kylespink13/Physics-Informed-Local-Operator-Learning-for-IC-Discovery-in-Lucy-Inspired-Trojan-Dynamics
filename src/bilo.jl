# Initialize test region
struct TrustRegion
    center::Vector{Float64}
    radius::Vector{Float64}
end

sample_x0(tr::TrustRegion, rng::AbstractRNG, n::Int) =
    tr.center .+ tr.radius .* (2 .* rand(rng, 6, n) .- 1)

function bilo_run(op::LocalOperator, ps, st,
                  x0_init::AbstractVector, tr::TrustRegion,
                  t_grid::AbstractVector;
                  rng::AbstractRNG = Random.default_rng(),
                  n_outer::Int = 200,
                  n_cycles::Int = 8,
                  n_refit::Int = 10,
                  n_inner::Int = 4,
                  lr_θ = 1e-3,
                  lr_x0 = 1e-2,
                  target::Symbol = :tadpole,
                  mu = MU_SUN_JUPITER,
                  λ_J = 0.1,
                  verbose::Bool = true)
    x0 = collect(float.(x0_init))

    opt_θ  = Optimisers.setup(Optimisers.Adam(lr_θ),  ps)
    opt_x0 = Optimisers.setup(Optimisers.Adam(lr_x0), x0)

    history = NamedTuple[]
    surrogate(x) = first(forward_grid(op, ps, st, x, t_grid))

    outer_per_cycle = div(n_outer, n_cycles)
    step = 0

    for c in 1:n_cycles
        l_in = 0.0
        for _ in 1:n_refit
            x0_batch = sample_x0(tr, rng, n_inner)
            l_in, gθ = Zygote.withgradient(
                p -> supervised_loss(op, p, st, x0_batch, t_grid; mu=mu, λ_J=λ_J), ps
            )
            opt_θ, ps = Optimisers.update!(opt_θ, ps, gθ[1])
        end

        l_out = NaN
        for _ in 1:outer_per_cycle
            step += 1
            l_out, gx = Zygote.withgradient(
                x -> outer_loss(surrogate, x, t_grid; target=target, mu=mu), x0
            )
            opt_x0, x0 = Optimisers.update!(opt_x0, x0, gx[1])

            @inbounds for i in eachindex(x0)
                lo, hi = tr.center[i] - tr.radius[i], tr.center[i] + tr.radius[i]
                x0[i] = clamp(x0[i], lo, hi)
            end

            push!(history, (; step=step, inner=l_in, outer=l_out, x0=copy(x0)))
        end

        tr = TrustRegion(copy(x0), tr.radius)

        verbose &&
            @printf "  cycle %2d   step %4d   L_inner = %.3e   L_outer = %.3e\n" c step l_in l_out
    end
    return ps, st, x0, history, tr
end

function direct_search(x0_init::AbstractVector, t_grid::AbstractVector;
                       n_steps::Int = 200,
                       lr = 1e-2,
                       target::Symbol = :tadpole,
                       mu = MU_SUN_JUPITER,
                       verbose::Bool = true)
    x0  = collect(float.(x0_init))
    opt = Optimisers.setup(Optimisers.Adam(lr), x0)
    propagator(x) = propagate_diff(x, t_grid; mu=mu)
    history = NamedTuple[]

    for k in 1:n_steps
        l, g = Zygote.withgradient(
            x -> outer_loss(propagator, x, t_grid; target=target, mu=mu), x0
        )
        opt, x0 = Optimisers.update!(opt, x0, g[1])
        push!(history, (; step=k, outer=l, x0=copy(x0)))
        verbose && k % 25 == 0 &&
            @printf "  step %4d   L_outer = %.3e\n" k l
    end
    return x0, history
end
