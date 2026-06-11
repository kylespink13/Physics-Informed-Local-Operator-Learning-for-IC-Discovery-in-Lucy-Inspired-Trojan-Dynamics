@enum TrajectoryClass TADPOLE HORSESHOE ESCAPE BOUND

function libration_amplitude(sol; mu=MU_SUN_JUPITER)
    j_pos = SVector(1 - mu, 0.0, 0.0)
    θ_min = +Inf
    θ_max = -Inf
    for u in sol.u
        r   = SVector(u[1], u[2], u[3])
        rel = r .- j_pos
        θ   = atan(rel[2], rel[1])
        θ_min = min(θ_min, θ)
        θ_max = max(θ_max, θ)
    end
    return θ_max - θ_min
end

function classify_trajectory(sol; mu=MU_SUN_JUPITER, escape_radius=5.0)
    for u in sol.u
        norm(SVector(u[1], u[2], u[3])) > escape_radius && return ESCAPE
    end
    swing = libration_amplitude(sol; mu=mu)
    swing > deg2rad(220) && return HORSESHOE #criteria for horseshoe orbit
    swing < deg2rad(180) && return TADPOLE #criteria for tadpole orbit
    return BOUND
end
