function supervised_loss(op::LocalOperator, ps, st,
                         x0_batch::AbstractMatrix, t_grid::AbstractVector;
                         mu=MU_SUN_JUPITER, λ_J=0.1)
    N  = size(x0_batch, 2)
    T  = length(t_grid)
    Xs = Zygote.ignore() do
        [propagate_diff(x0_batch[:, k], t_grid; mu=mu) for k in 1:N]
    end
    L = 0.0
    for k in 1:N
        x0   = x0_batch[:, k]
        Y, _ = forward_grid(op, ps, st, x0, t_grid)

        L += sum(abs2, Y .- Xs[k]) / T

        if λ_J > 0
            J0 = jacobi_constant(x0, mu)
            jac_pen = 0.0
            for j in 1:T
                jac_pen += (jacobi_constant(Y[:, j], mu) - J0)^2
            end
            L += λ_J * jac_pen / T
        end
    end
    return L / N
end

function outer_loss(propagator, x0::AbstractVector, t_grid::AbstractVector;
                    target::Symbol=:tadpole,
                    mu=MU_SUN_JUPITER,
                    target_amplitude::Real=NaN,
                    close_weight=1.0)
    X = propagator(x0)
    L4_pos = SVector(0.5 - mu, sqrt(3)/2, 0.0)
    j_pos  = SVector(1 - mu, 0.0, 0.0)

    amp_sq = 0.0
    close_pen = 0.0
    for j in 1:size(X, 2)
        r = SVector(X[1, j], X[2, j], X[3, j])
        amp_sq = max(amp_sq, sum(abs2, r .- L4_pos))
        close_pen += exp(-10 * norm(r .- j_pos))
    end
    amp = sqrt(amp_sq + eps())

    target_amp = !isnan(target_amplitude) ? float(target_amplitude) :
                 target === :tadpole ? 0.05 :
                 target === :horseshoe ? 0.30 :
                 error("Unknown outer target: $target")

    return (amp - target_amp)^2 + close_weight * close_pen / size(X, 2)
end
