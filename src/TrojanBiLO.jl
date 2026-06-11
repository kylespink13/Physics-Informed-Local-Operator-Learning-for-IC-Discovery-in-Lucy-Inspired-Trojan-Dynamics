module TrojanBiLO

using LinearAlgebra
using Statistics
using Random
using Printf
using StaticArrays

using OrdinaryDiffEq
using SciMLSensitivity
using Zygote
using ForwardDiff

using Lux
using Optimisers
using ComponentArrays

include("cr3bp.jl")
include("classifier.jl")
include("operator.jl")
include("losses.jl")
include("bilo.jl")
include("plotting.jl")  


export MU_SUN_JUPITER
export cr3bp_rhs, jacobi_constant, l4_position, l5_position
export propagate_cr3bp, propagate_diff

export TrajectoryClass, TADPOLE, HORSESHOE, ESCAPE, BOUND
export classify_trajectory, libration_amplitude

export LocalOperator, build_operator, forward, forward_grid
export init_params

export supervised_loss, outer_loss

export TrustRegion, sample_x0
export bilo_run, direct_search

export plot_trajectory!, plot_loss_curve!, plot_jacobi_drift!

end # module
