struct LocalOperator{B,T,R,S}
    branch::B
    trunk::T
    p_dim::Int
    x_ref::R
    x_scale::S # per-component scale for (x0 - x_ref)
    t_scale::Float64 
end

function build_operator(x_ref::AbstractVector;
                        p_dim::Int=32,
                        branch_hidden::Vector{Int}=[32, 32, 32],
                        trunk_hidden::Vector{Int}=[32, 32, 32],
                        t_scale::Real=10.0,
                        x_scale::AbstractVector=fill(1.0, length(x_ref)))
    branch_layers = Any[]
    push!(branch_layers, Dense(6, branch_hidden[1], tanh))
    for i in 1:length(branch_hidden)-1
        push!(branch_layers, Dense(branch_hidden[i], branch_hidden[i+1], tanh))
    end
    push!(branch_layers, Dense(branch_hidden[end], 6 * p_dim))
    branch = Chain(branch_layers...)

    trunk_layers = Any[]
    push!(trunk_layers, Dense(1, trunk_hidden[1], tanh))
    for i in 1:length(trunk_hidden)-1
        push!(trunk_layers, Dense(trunk_hidden[i], trunk_hidden[i+1], tanh))
    end
    push!(trunk_layers, Dense(trunk_hidden[end], p_dim))
    trunk = Chain(trunk_layers...)

    xs = [s == 0 ? 1.0 : float(s) for s in x_scale]
    return LocalOperator(branch, trunk, p_dim, collect(float.(x_ref)), xs, float(t_scale))
end

function init_params(op::LocalOperator, rng::AbstractRNG)
    ps_b, st_b = Lux.setup(rng, op.branch)
    ps_t, st_t = Lux.setup(rng, op.trunk)
    ps = (branch = ps_b, trunk = ps_t)
    st = (branch = st_b, trunk = st_t)
    return ps, st
end

function forward(op::LocalOperator, ps, st, x0::AbstractVector, t::Real)
    x_in = (x0 .- op.x_ref) ./ op.x_scale
    b, st_b = op.branch(x_in, ps.branch, st.branch)
    τ, st_t = op.trunk([float(t) / op.t_scale], ps.trunk, st.trunk)
    B = reshape(b, 6, op.p_dim)
    return op.x_ref .+ B * τ, (branch=st_b, trunk=st_t)
end

function forward_grid(op::LocalOperator, ps, st, x0::AbstractVector,
                      t_grid::AbstractVector)
    x_in = (x0 .- op.x_ref) ./ op.x_scale
    b, st_b = op.branch(x_in, ps.branch, st.branch)
    B = reshape(b, 6, op.p_dim)
    T_in = reshape(collect(float.(t_grid)) ./ op.t_scale, 1, length(t_grid))
    τ_all, st_t = op.trunk(T_in, ps.trunk, st.trunk)
    Y = op.x_ref .+ B * τ_all
    return Y, (branch=st_b, trunk=st_t)
end
