using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using TrojanBiLO
using JLD2
using Random
using ProgressMeter
using Printf

const MU       = MU_SUN_JUPITER
const T_FINAL  = 50.0
const N_SAMPLES = 400
const RADIUS   = [0.02, 0.02, 0.0, 0.01, 0.01, 0.0]

rng     = MersenneTwister(0) #random number generator
L4      = l4_position(MU)
center  = vcat(collect(L4), [0.0, 0.0, 0.0])

trajectories = Vector{Any}(undef, N_SAMPLES)
labels       = Vector{TrajectoryClass}(undef, N_SAMPLES)
x0_all       = zeros(6, N_SAMPLES) #initial state

@info "propagating $N_SAMPLES trajectories near L4 over 0 < t < $T_FINAL"
@showprogress for k in 1:N_SAMPLES
    x0  = center .+ RADIUS .* (2 .* rand(rng, 6) .- 1)
    sol = propagate_cr3bp(x0, (0.0, T_FINAL); mu=MU, saveat=0.05)
    x0_all[:, k]      = x0
    trajectories[k]   = sol
    labels[k]         = classify_trajectory(sol; mu=MU)
end

n_tad = count(==(TADPOLE),   labels)
n_hs  = count(==(HORSESHOE), labels)
n_esc = count(==(ESCAPE),    labels)
n_bd  = count(==(BOUND),     labels)
@printf "\nlabels:  tadpole = %d   horseshoe = %d   escape = %d   bound = %d\n" n_tad n_hs n_esc n_bd

#save trajectory label dataset
mkpath(joinpath(@__DIR__, "..", "data"))
outfile = joinpath(@__DIR__, "..", "data", "monte_carlo_benchmark.jld2")
@save outfile x0_all trajectories labels
@info "saved dataset" outfile
