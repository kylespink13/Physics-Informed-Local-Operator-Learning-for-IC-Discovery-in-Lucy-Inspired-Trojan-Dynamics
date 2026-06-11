const MU_SUN_JUPITER = 9.5388e-4 #non-dimensionalized

l4_position(mu=MU_SUN_JUPITER) = SVector(0.5 - mu,  sqrt(3)/2, 0.0)
l5_position(mu=MU_SUN_JUPITER) = SVector(0.5 - mu, -sqrt(3)/2, 0.0)

function cr3bp_rhs(state, mu, t)
    x, y, z, vx, vy, vz = state
    r1 = sqrt((x + mu)^2 + y^2 + z^2)
    r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2)
    ax = 2vy + x - (1 - mu) * (x + mu) / r1^3 - mu * (x - 1 + mu) / r2^3 #x-acceleration
    ay = -2vx + y - (1 - mu) * y / r1^3 - mu * y / r2^3  #y-acceleration
    az = - (1 - mu) * z / r1^3 - mu * z / r2^3 #z-acceleration
    return SVector(vx, vy, vz, ax, ay, az)
end

function jacobi_constant(state, mu=MU_SUN_JUPITER)
    x, y, z, vx, vy, vz = state
    r1 = sqrt((x + mu)^2     + y^2 + z^2)
    r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2)
    U  = (x^2 + y^2) + 2(1 - mu)/r1 + 2mu/r2
    return U - (vx^2 + vy^2 + vz^2)
end

function propagate_cr3bp(x0::AbstractVector, tspan;
                         mu=MU_SUN_JUPITER, reltol=1e-12, abstol=1e-12,
                         saveat=nothing)
    prob = ODEProblem((u, p, t) -> cr3bp_rhs(u, p, t), SVector{6}(x0), tspan, mu)
    return solve(prob, Vern9();
                 reltol=reltol, abstol=abstol,
                 saveat=isnothing(saveat) ? eltype(tspan)[] : saveat)
end

function propagate_diff(x0::AbstractVector, t_grid::AbstractVector;
                        mu=MU_SUN_JUPITER,
                        reltol=1e-9, abstol=1e-9,
                        sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP(allow_nothing=true)))
    tspan = (float(first(t_grid)), float(last(t_grid)))
    u0    = collect(float.(x0))
    rhs!  = (du, u, p, t) -> (du .= cr3bp_rhs(u, mu, t))
    prob  = ODEProblem(rhs!, u0, tspan)
    sol   = solve(prob, Vern9();
                  reltol=reltol, abstol=abstol,
                  saveat=t_grid, sensealg=sensealg)
    return Array(sol)
end
