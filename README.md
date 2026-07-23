[![CI](https://github.com/org-arl/DataDrivenAcoustics.jl/workflows/CI/badge.svg)](https://github.com/org-arl/DataDrivenAcoustics.jl/actions)

# DataDrivenAcoustics

# Ray Models

This package is built upon the ideas discussed in our journal paper "Data-Aided Underwater Acoustic Ray Propagation Modeling" published on IEEE Journal of Oceanic Engineering (available [online](https://ieeexplore.ieee.org/abstract/document/10224658)). It provides a Ray-basis neural network implementation for use with [`UnderwaterAcoustics.jl`](https://github.com/org-arl/UnderwaterAcoustics.jl).

Conventional acoustic propagation models require accurate environmental knowledge to be available beforehand. While data-driven techniques might allow us to model acoustic propagation without the need for extensive prior environmental knowledge, such techniques tend to be data-hungry. We propose a physics-based data-driven acoustic propagation modeling approach that enables us to train models with only a small amount of data. The proposed modeling framework is not only data-efficient, but also offers flexibility to incorporate varying degrees of environmental knowledge, and generalizes well to permit extrapolation beyond the area where data were collected.

> [!NOTE]
> The API for `DataDrivenAcoustics.jl` changed significantly in `v0.3` to align itself with newer versions of `UnderwaterAcoustics.jl`. Some of the functionality from `v0.2` has not yet been ported to `v0.3`, so if you need older functionality, please use `v0.2`. We will add back much of the functionality and more soon!

## Installation

```julia
julia> # press ]
pkg> add UnderwaterAcoustics, DataDrivenAcoustics
```

## Usage

We first start by loading some helpful dependencies:
```julia
using UnderwaterAcoustics
using DataDrivenAcoustics
using StableRNGs
using Plots
```
and then prepare a dataset by sampling transmission loss at 1000 random locations from a `PekerisRayTracer` propagation model:
```julia
env = UnderwaterEnvironment(seabed=Rock, bathymetry=200.0)
pm1 = PekerisRayTracer(env; max_bounces=3)
tx = AcousticSource(0, -11, 250)
rxpos = rand(StableRNG(27), 2, 1000) .* [200.0, 40.0] .+ [5500.0, -110.0]
rxs = [AcousticReceiver(rxpos[1,i], rxpos[2,i]) for i ∈ 1:size(rxpos,2)]
xloss = Float32.(transmission_loss(pm1, tx, rxs))
```
We use a `StableRNG` random number generator for reproducibility of this example. We now have transmission loss data measured at 1000 random locations in a 5.5 to 5.7 km range and 70 to 110 m depth.

We would like to use this data to build a data-driven propagation model:
```julia
pm = DataDrivenPropagationModel(RayBasisNN_2D(60); rng=StableRNG(42))
```
This creates an untrained model, initialized with random weights. We next prepare a loss function that measures the prediction error for the dataset we created:
```julia
rxs = [AcousticReceiver(x, z) for (x, z) ∈ zip(rxpos[1,:], rxpos[2,:])]
loss = TransmissionLossMSE(pm, AcousticSource(nothing, 250), rxs, xloss)
```
Once we have the loss function, we can train the propagation model. We do the training in 2-phases, as is common for physics-guided problems. The first phase uses an `Adam` optimizer to find a good solution:
```julia
DataDrivenAcoustics.fit!(pm, loss;
  optimizer = Adam(5e-6),           # ADAM with specified learning rate
  minloss = 100,                    # minimize until loss < 100
  maxiters = 5000,                  # or until 5000 epochs have passed
  show_progress = 100)              # print progress every 100 epochs
```
The second phase uses a `BFGS` optimizer to refine the solution to a local minimum:
```julia
DataDrivenAcoustics.fit!(pm, loss;
  optimizer = BFGS(),               # BFGS quasi-Newton optimizer
  maxiters = 200,                   # minimize to a maximum of 200 iterations
  show_progress = 1)                # print progress every iteration
```
We can now use the model to predict transmission loss in an area of interest. Note that the model is able to extrapolate well beyond the area where measurements were made (shown as block dots below):
```julia
rx = AcousticReceiverGrid2D(5300:6000, -200:-20)
x = transmission_loss(pm, AcousticSource(nothing, 250), rx)
plot(rx, x; clim=(50,100), xlims=(5300,6000), ylims=(-200,-20))
scatter!([p for p ∈ zip(rxpos[1,:], rxpos[2,:])]; markersize=0.5, color=:black)
```
![](docs/images/ex1.png)

We compare this with the ground truth from the original physics-based propagation model:
```julia
x = transmission_loss(pm1, tx, rx)
plot(rx, x; clim=(50,100), xlims=(5300,6000), ylims=(-200,-20))
scatter!([p for p ∈ zip(rxpos[1,:], rxpos[2,:])]; markersize=0.5, color=:black)
```
![](docs/images/ex1-gt.png)

While we see that the match is not perfect, it is pretty impressive given that we have no measurements in the extrapolated area!

## Publications
### Primary paper

- K. Li and M. Chitre, “Data-aided underwater acoustic ray propagation modeling,” 2023. [(online)](https://ieeexplore.ieee.org/abstract/document/10224658)

### Other useful papers

- K. Li and M. Chitre, “Ocean acoustic propagation modeling using scientific machine learning,” in OCEANS: San Diego–Porto. IEEE, 2021, pp. 1–5.
- K. Li and M. Chitre, “Physics-aided data-driven modal ocean acoustic propagation modeling,” in International Congress of Acoustics, 2022.


# Modal Models
# MBNN
 
A physics-informed neural network for predicting underwater acoustic fields
and inverting sound-speed profiles in shallow water, from sparse amplitude
measurements.
 
## What the model does
 
`ModalBasis` predicts the complex acoustic pressure p(r, z) at any range
and depth in a 2D range-independent waveguide. Instead of a generic black-box
network, it is built directly on normal-mode theory: the field is a sum of
modes, each with a horizontal wavenumber kᵣ and a depth function ψ(z)
constructed from phase integrals of the local wavenumber
k(z) = ω / c(z). Because the structure *is* the physics, the model needs far
less data than a plain neural network and extrapolates outside the measured
region instead of falling apart there.
 
The twist is that the sound-speed profile c(z) is not assumed known. A small
sound-speed network (SSNN — one input, one hidden ReLU layer) represents
c(z), bounded to a physical range [cmin, cmax]. Everything is trained jointly
by gradient descent (Zygote) against measured field amplitudes:
 
- **A, B** — complex modal coefficients (one pair per mode)
- **kᵣ** — horizontal wavenumbers, bounded to the physically
  admissible band and warm-started from a Pekeris mode solver
- **SSNN weights** — the sound-speed profile itself

So a single training run gives you two things at once: a field predictor and
an inverted c(z).
 
## What it can be used for
 
- **Field prediction beyond the measurements** — train on amplitudes from a
  narrow range band (e.g. a 50 m strip of receiver positions) and predict the
  field over a much wider area of interest.
- **Sound-speed inversion** — recover the full c(z) profile from field data
  plus just a few direct shallow measurements (e.g. five CTD points in the
  top 4 m), without ever measuring the deep profile.
- **Modal structure** — read off the learned per-mode wavenumbers and the
  mode shapes implied by the learned c(z).
## Quick start

```julia
include("MBNN.jl")
using .ModalSolverPaper
using Lux, Random
 
model = ModalBasisNN_2D(25.0f0, 500.0f0; nmodes = 6, nhidden = 12)  # 25 m waveguide, 500 Hz
ps, st = Lux.setup(MersenneTwister(42), model)   # Pekeris warm start happens here
 
X = Float32[675.0 680.0 690.0;                   # row 1: range (m)
              5.0  10.0  20.0]                   # row 2: depth (m)
amp = amplitude_output(model, ps, st, X)         # pressure amplitude |p|
c   = sound_speed_grid(model, ps)                # learned c(z) on the depth grid
kr  = horizontal_wavenumbers(model, ps)          # learned kᵣ per mode
```
 
Untrained parameters give a physically valid but arbitrary field — the point
is to fit `ps` to measurements, which is what the rest of this README does.

# Quick start output

Running the above should produce something close to:

```julia
julia> amp = amplitude_output(model, ps, st, X)
3-element Vector{Float32}:
 0.004187098
 0.002502838
 0.0010821978

julia> c = sound_speed_grid(model, ps)
201-element Vector{Float32}:
 1451.2946
 1451.2817
    ⋮
 1449.2657

julia> kr = horizontal_wavenumbers(model, ps)
6-element Vector{Float32}:
 2.1633189
 2.1533737
 2.1366236
 2.1128173
 2.081619
 2.0426033
```

Exact numbers will differ if you change the RNG seed, `nmodes`, `nhidden`,
or the environment settings.

For a complete example of joint field prediction and sound speed inversion, please see examples.

## Publications

### Primary paper

- K. Li and M. Chitre, "Physics-aided data-driven modal ocean acoustic
  propagation modeling," in International Congress of Acoustics, 2022.

### Other useful papers

- K. Li and M. Chitre, "Data-aided underwater acoustic ray propagation
  modeling," 2023.
- F. B. Jensen, W. A. Kuperman, M. B. Porter, and H. Schmidt, *Computational
  Ocean Acoustics*, 2nd ed. New York: Springer, 2011.
