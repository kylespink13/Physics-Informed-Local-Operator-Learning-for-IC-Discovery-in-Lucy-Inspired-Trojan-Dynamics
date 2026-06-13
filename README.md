# TrojanBiLO

Solver-in-the-loop bilevel local operator (BiLO) learning for
initial-condition discovery in Lucy-inspired Trojan dynamics.

The method studies the natural CR3BP dynamics in the Sun–Jupiter L4/L5 co-orbital region — where Jupiter's
Trojan asteroids (NASA's Lucy mission targets) live — and asks which
initial conditions produce a prescribed qualitative behavior (tadpole,
horseshoe, escape).

Hybrid framework:

- **BiLO** (Zhang & Lowengrub, 2024) — outer trust-region bilevel
  structure. A local flow-map surrogate is fit cheaply over a
  neighborhood of the current iterate and the inverse search is carried
  out through it.
- **Solver-in-the-loop** (Um et al., 2020) — inner model. Instead of a
  PDE-residual loss, the surrogate is trained supervised against a
  differentiable numerical integrator (`SciMLSensitivity` + `Vern9`).
  For a 6-D ODE this avoids the pathological residual-loss landscape of
  PINNs near separatrices and is the natural use of the Julia SciML
  stack.

A pure differentiable-physics baseline (`direct_search`) is included for
comparison.

## Layout

```
TrojanBiLO/
├── Project.toml
├── src/
│   ├── TrojanBiLO.jl         module entry
│   ├── cr3bp.jl              dynamics, Jacobi, propagate_cr3bp/diff
│   ├── classifier.jl         tadpole / horseshoe / escape labeling
│   ├── operator.jl           branch–trunk LocalOperator surrogate
│   ├── losses.jl             supervised + outer (target) losses
│   ├── bilo.jl               TrustRegion, bilo_run, direct_search
│   └── plotting.jl           GLMakie helpers (Makie.inline!(true))
├── scripts/
│   ├── 01_generate_data.jl       Monte Carlo benchmark
│   ├── 02_train_local_operator.jl    surrogate sanity check
│   ├── 03_bilevel_search.jl      BiLO + direct baseline
│   └── 04_validate_and_plot.jl   long-horizon Vern9 + figures
├── test/runtests.jl
├── data/                     gitignored — generated datasets
└── figures/                  gitignored — generated plots
```

## Setup

```julia
julia> ]
(@v1.x) pkg> activate .
(TrojanBiLO) pkg> instantiate
```

## Run

Each script self-activates the project; no special shell needed:

```bash
julia scripts/01_generate_data.jl
julia scripts/02_train_local_operator.jl
julia scripts/03_bilevel_search.jl
julia scripts/04_validate_and_plot.jl
```

Expected wall time on a recent laptop (CPU only): ≲5 min per script.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
