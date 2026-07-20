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
include("ModalSolver_paper_ssnn.jl")
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

# Sample Implementation

This walks through the entire example chunk by chunk. Paste each block into
the Julia REPL **in order** and it produces a sample plot of ground truth generated by Kraken.

## 1. Load packages

```julia
using CSV, DataFrames, Random, Statistics, LinearAlgebra, Plots
using Lux, Zygote, Optimisers, ProgressMeter
using UnderwaterAcoustics, AcousticsToolbox

gr() #plots backend
```

## 2. Load the modal solver

The MBNN + SSNN model lives in its own module. This makes `ModalBasisNN_2D`,
`sound_speed_grid`, `horizontal_wavenumbers`, `amplitude_output`, and
`depth_interpolation_matrix` available:

```julia
include(joinpath(@__DIR__, "ModalSolver_paper_ssnn.jl"))
using .ModalSolverPaper
```

## 3. File names

Every file the example touches is declared up front as a constant, and all of
them live next to the script (`@__DIR__`). They fall into four groups:

**Input (must exist before running).**
`REFERENCE_SSP_FILE` is the ground-truth sound-speed profile. It can be 
generated by Kraken or given by the user.

**Training outputs (created by steps 11–14).**
`HISTORY_FILE` logs one row per checkpoint per restart: `restart`, `epoch`,
`train_objective`, `train_field_mse`, `val_field_mse`.`PARAMETER_FILE` is 
every trained parameter of the best run, flattened into `parameter`/`value`
pairs (e.g.`A_re_1 … A_re_6`, `ssp_W1_1 … ssp_W1_12`), so a run can be
reloaded or inspected without Julia.

**Plots (created by steps 13–15).**
`SSP_PLOT_FILE` overlays reference vs. learned SSP with the five anchor
measurements; `AOI_PLOT_FILE` shows Kraken ground truth and model prediction
side by side as dB heatmaps with the training band marked; `HISTORY_PLOT_FILE`
is the train/validation MSE curve of the best restart.

Finally, `FORCE_REGENERATE_GROUND_TRUTH` controls whether step 7 reruns
Kraken. Leave it `true` for the very first run so the CSVs are guaranteed to
match this exact setup (depth, frequency, seabed, grids); afterwards set it
to `false` — the Kraken sweep over the 0.1 m AOI grid is by far the slowest
part of the pipeline, and regenerating also redraws the train/validation
split (fresh entropy), which changes results run to run.

```julia
#input
const REFERENCE_SSP_FILE = "reference_ssp_100pts.csv"
#kraken-generated data
const MEASUREMENT_FILE   = "ssnn_profiles_1224.csv"
const AOI_FILE           = "ssnn_ground_truth_aoi.csv"
#training outputs
const HISTORY_FILE       = "paper_ssnn_training_history.csv"
const PARAMETER_FILE     = "paper_ssnn_parameters.csv"
const SSP_PRED_FILE      = "paper_ssnn_predicted_ssp.csv"
const AOI_PRED_FILE      = "paper_ssnn_aoi_prediction.csv"
#plots
const SSP_PLOT_FILE      = "paper_ssnn_true_vs_predicted_ssp.png"
const AOI_PLOT_FILE      = "paper_ssnn_true_vs_predicted_aoi.png"
const HISTORY_PLOT_FILE  = "paper_ssnn_training_history.png"

# Regenerate once so the Kraken data definitely matches this setup.
#change to false after the first successful run to avoid recomputing it
const FORCE_REGENERATE_GROUND_TRUTH = true
```

## 4. Settings

Environment, model size, training budget, and loss weights. The loss is the
paper's eqn. 22 (field error + L1 on the modal coefficients) plus an SSP
anchor term (five shallow measurements, Section III-B / Fig. 10) and a soft
prior that total SSP variation stays below 35 m/s:

```julia
const D = 25.0f0;       const FREQ = 500.0f0;   const SOURCE_DEPTH = 5.0
const NMODES = 6;       const NHIDDEN = 12;     const NGRID = 201
#sound speed bounds plus the initial guess so the model doesnt go haywire
const CMIN = 1400.0f0;  const CMAX = 1500.0f0;  const CINIT = 1450.0f0
const MAX_SSP_SPAN = 35.0f0
const RREF = 675.0 #reference range inside the measurement band
#random restarts escape bad initializations; early stopping checks every LOG_EVERY epochs
const RESTARTS = 10;    const EPOCHS = 8000;    const BATCH_SIZE = 256
const LOG_EVERY = 200;  const EARLY_STOPPING_CHECKS = 12
const LEARNING_RATE = 1.0f-3;  const GRAD_CLIP = 100.0

# eqn 22 loss function: field error + L1(A) + L1(B)
const ALPHA = 1.0f-6;  const BETA = 1.0f-6

# Section III-B / Fig. 10: shallow SSP measurements at 0, 1, 2, 3, and 4 m.
const SSP_ANCHOR_DEPTHS = Float32[0, 1, 2, 3, 4]

# the SSP error is normalized by 35 m/s before applying this weight
const LAMBDA_SSP = 20.0f0
const LAMBDA_SPAN = 1.0f0
```

## 5. Helpers

Small utilities used throughout — flattening parameter trees, building the
2 × N model input, the smoothed complex L1, and the reference-SSP loader
(`SampledField` acts as a 1-D linear interpolator on depth):

```julia
save_plot(plt, f) = (display(plt); savefig(plt, joinpath(@__DIR__, f)); println("Saved: ", joinpath(@__DIR__, f)))
flat(x)          = Optimisers.destructure(x)[1]   # numeric array leaves of any param/grad tree
allfinite(x)     = all(isfinite, flat(x))
make_input(r, d)   = Float32.(vcat(r', d'))       # 2 x N (range; depth)
l1_complex(re, im) = sum(sqrt.(re .^ 2 .+ im .^ 2 .+ 1f-12)) #smoothed so the gradient exists at 0

function load_reference_ssp()
    df = sort!(CSV.read(joinpath(@__DIR__, REFERENCE_SSP_FILE), DataFrame), :depth_m)
    z, c = Float64.(df.depth_m), Float64.(df.c_ms)
    return z, c, SampledField(c; z = z, interp = Linear())
end
```

## 6. Kraken data generation

Defines the synthetic ground truth: one Kraken run on a receiver grid
(`sample_field`), full generation of the AOI grid plus the 1,224 measurement
points with a 70/30 train/validation split (`generate_ground_truth`), and a
guard that only regenerates when needed (`ensure_ground_truth`):

```julia
function sample_field(pm, tx, ranges, depths)
    amp = abs.(acoustic_field(pm, tx, AcousticReceiverGrid2D(ranges, -depths)))
    @assert size(amp) == (length(ranges), length(depths)) "Unexpected acoustic field size: $(size(amp))"
    a = vec(permutedims(amp))   # depth fastest, then range
    DataFrame(
        range_m = repeat(collect(ranges); inner = length(depths)),
        depth_m = repeat(collect(depths); outer = length(ranges)),
        amp = a,
        db = 20 .* log10.(a .+ 1e-30), #tiny offset avoids log10(0)
    )
end

function generate_ground_truth()
    Random.seed!() #fresh entropy, so the train/validation split differs run to run
    z_ref, c_ref, _ = load_reference_ssp()
    println("Generating Kraken data...")

    env = UnderwaterEnvironment(
        bathymetry = Float64(D),
        soundspeed = SampledField(c_ref;
            z = range(0.0, -Float64(D); length = length(c_ref)), interp = CubicSpline()),
        seabed = SandyClay,
    )
    pm = Kraken(env; rmax = 850.0, nmodes = NMODES)
    tx = AcousticSource(0.0, 0.0, -SOURCE_DEPTH, Float64(FREQ); spl = 0.0)

    #dense 0.1 m grid over the whole area of interest, only used for evaluation later
    CSV.write(joinpath(@__DIR__, AOI_FILE), sample_field(pm, tx, 550.0:0.1:800.0, 1.0:0.1:24.0))

    #51 ranges x 24 depths = 1224 synthetic measurements inside the 650-700 m band
    meas = sample_field(pm, tx, 650.0:1.0:700.0, 1.0:1.0:24.0)
    split = fill("validation", nrow(meas))
    split[randperm(nrow(meas))[1:round(Int, 0.70 * nrow(meas))]] .= "train" #70/30 split
    meas.split = split
    CSV.write(joinpath(@__DIR__, MEASUREMENT_FILE), meas)
end

function ensure_ground_truth()
    have = isfile(joinpath(@__DIR__, MEASUREMENT_FILE)) && isfile(joinpath(@__DIR__, AOI_FILE))
    if FORCE_REGENERATE_GROUND_TRUTH || !have
        generate_ground_truth()
    else
        println("Using existing Kraken CSV files.")
    end
end
```

## 7. Generate the data

Run it:

```julia
ensure_ground_truth()
z_ref, c_ref, ref_ssp = load_reference_ssp()
```

## 8. Load the measurements and build the training inputs

The train/validation split comes from the CSV itself; amplitudes are
normalized by the mean training amplitude:

```julia
df = CSV.read(joinpath(@__DIR__, MEASUREMENT_FILE), DataFrame)
s = lowercase.(String.(df.split))
train_df = df[s .== "train", :]
val_df   = df[(s .== "validation") .| (s .== "val"), :]
isempty(train_df) && error("No training rows found")
isempty(val_df)   && error("No validation rows found")

X_train = make_input(train_df.range_m, train_df.depth_m)
X_val   = make_input(val_df.range_m, val_df.depth_m)
yscale  = mean(Float64.(train_df.amp))
yscale > 0 || error("Mean training amplitude must be positive")
y_train = Float32.(train_df.amp ./ yscale)
y_val   = Float32.(val_df.amp ./ yscale)
```

## 9. Build the model and SSP anchors

Construct the MBNN and the fixed interpolation matrix that reads the model's
c(z) at the five anchor depths; users are free to edit the parameters:

```julia
model = ModalBasisNN_2D(D, FREQ; nmodes = NMODES, nhidden = NHIDDEN,
                        cmin = CMIN, cmax = CMAX, cinit = CINIT, ngrid = NGRID, rref = RREF)

# Five shallow CTD measurements, as in Section III-B / Fig. 10.
c_anchor_true = Float32.(ref_ssp.(Float64.(SSP_ANCHOR_DEPTHS)))
W_anchor = depth_interpolation_matrix(model, SSP_ANCHOR_DEPTHS)
```

## 10. Define the loss

Eqn. 22 pieces plus the two SSP terms, computed once and reused for logging:

```julia
function components(ps, st, X, y)
    amp = amplitude_output(model, ps, st, X) ./ Float32(yscale)
    cgrid = sound_speed_grid(model, ps)
    span_excess = max(maximum(cgrid) - minimum(cgrid) - MAX_SSP_SPAN, 0f0)
    return (
        field = mean(abs2, amp .- y),
        A = l1_complex(ps.A_re, ps.A_im),
        B = l1_complex(ps.B_re, ps.B_im),
        ssp = mean(abs2, (W_anchor * cgrid .- c_anchor_true) ./ MAX_SSP_SPAN),
        span = (span_excess / MAX_SSP_SPAN)^2,
    )
end
total(c) = c.field + ALPHA * c.A + BETA * c.B + LAMBDA_SSP * c.ssp + LAMBDA_SPAN * c.span
objective(ps, st, X, y) = total(components(ps, st, X, y))
```

## 11. Train

Random restarts with minibatch Adam, gradient clipping, NaN guards, periodic
full-dataset logging, and early stopping on the validation field MSE. The
best restart overall is kept in `best`:

```julia
best = (val = Inf, ps = nothing, st = nothing, restart = 0, epoch = 0)
history = DataFrame(restart = Int[], epoch = Int[], train_objective = Float64[],
                    train_field_mse = Float64[], val_field_mse = Float64[],
                    anchor_rmse_ms = Float64[], ssp_min_ms = Float64[], ssp_max_ms = Float64[],
                    grad_norm = Float64[])
ntrain = length(y_train)

for restart in 1:RESTARTS
    rng = MersenneTwister()
    ps, st = Lux.setup(rng, model)
    opt_state = Optimisers.setup(OptimiserChain(ClipNorm(GRAD_CLIP), Adam(LEARNING_RATE)), ps)

    rbest_val, rbest_ps, rbest_epoch, stale = Inf, deepcopy(ps), 0, 0
    pbar = Progress(EPOCHS; desc = "Restart $restart/$RESTARTS ", barlen = 28, showspeed = true)

    for epoch in 1:EPOCHS
        #fresh random minibatch every epoch
        idxs = rand(rng, 1:ntrain, min(BATCH_SIZE, ntrain))
        loss, grads = Zygote.withgradient(p -> objective(p, st, X_train[:, idxs], y_train[idxs]), ps)
        g = grads[1]
        #stop the restart instead of letting NaNs poison Adam's state
        if !isfinite(loss) || !allfinite(g)
            println("\nRestart $restart stopped: NaN/Inf at epoch $epoch"); break
        end
        opt_state, ps = Optimisers.update(opt_state, ps, g)
        if !allfinite(ps)
            println("\nRestart $restart stopped: invalid parameters"); break
        end

        #full-dataset metrics every LOG_EVERY epochs (plus first and last)
        if epoch == 1 || epoch % LOG_EVERY == 0 || epoch == EPOCHS
            train_c = components(ps, st, X_train, y_train)
            val_c   = components(ps, st, X_val, y_val)
            cgrid = Float64.(sound_speed_grid(model, ps))
            anchor_rmse = sqrt(mean((Float64.(W_anchor * Float32.(cgrid)) .- Float64.(c_anchor_true)) .^ 2))
            val_metric = Float64(val_c.field)
            push!(history, (restart, epoch, Float64(total(train_c)), Float64(train_c.field),
                            val_metric, anchor_rmse, minimum(cgrid), maximum(cgrid),
                            Float64(norm(flat(g)))))
            #early stopping tracks the validation field mse only
            if val_metric < rbest_val - 1e-10
                rbest_val, rbest_ps, rbest_epoch, stale = val_metric, deepcopy(ps), epoch, 0
            else
                stale += 1
            end
            next!(pbar; showvalues = [(:epoch, epoch),
                (:train_field, round(Float64(train_c.field); sigdigits = 5)),
                (:val_field, round(val_metric; sigdigits = 5)),
                (:anchor_rmse, round(anchor_rmse; sigdigits = 5)),
                (:ssp_span, round(maximum(cgrid) - minimum(cgrid); digits = 3))])
            if stale >= EARLY_STOPPING_CHECKS
                println("\nRestart $restart early-stopped at epoch $epoch"); break
            end
        else
            next!(pbar)
        end
    end

    #keep the best restart overall
    if rbest_val < best.val
        global best = (val = rbest_val, ps = rbest_ps, st = st, restart = restart, epoch = rbest_epoch)
    end
end

best.ps === nothing && error("All restarts failed")
ps, st = best.ps, best.st
CSV.write(joinpath(@__DIR__, HISTORY_FILE), history)
```

> Note: `global best` is needed when pasting the loop at top level in the
> REPL; inside the script's `main()` it is a plain local assignment.

## 12. Save the trained parameters

Every parameter flattened into one named CSV:

```julia
pname(s, n) = ["$(s)_$i" for i in 1:n]
par_names = vcat([pname(s, NMODES) for s in ("A_re", "A_im", "B_re", "B_im", "qkr")]...,
                 [pname(s, NHIDDEN) for s in ("ssp_W1", "ssp_b1", "ssp_W2")]..., "ssp_b2")
par_vals = Float64.(vcat(ps.A_re, ps.A_im, ps.B_re, ps.B_im, ps.qkr,
                         ps.ssp.W1, ps.ssp.b1, ps.ssp.W2, ps.ssp.b2))
CSV.write(joinpath(@__DIR__, PARAMETER_FILE), DataFrame(parameter = par_names, value = par_vals))
```

## 13. Evaluate the learned SSP

Compare the learned c(z) against the reference profile on the model depth
grid, save the CSV, and plot it with the five anchors:

```julia
z_pred = Float64.(model.ζ .* model.D) #model depth grid in metres
c_pred = Float64.(sound_speed_grid(model, ps))
c_true = Float64.(ref_ssp.(z_pred))
err = c_pred .- c_true
println("SSP RMSE = ", sqrt(mean(err .^ 2)), " m/s")
CSV.write(joinpath(@__DIR__, SSP_PRED_FILE),
          DataFrame(depth_m = z_pred, c_true_ms = c_true, c_pred_ms = c_pred, error_ms = err))

p_ssp = plot(c_ref, z_ref; yflip = true, lw = 2.5, label = "Reference SSP",
             xlabel = "Sound speed (m/s)", ylabel = "Depth (m)",
             title = "Paper-style SSNN inversion",
             xlims = (1400, 1500), xticks = 1400:25:1500, ylims = (0, 25), yticks = 0:5:25)
plot!(p_ssp, c_pred, z_pred; lw = 2.5, ls = :dash, label = "Learned SSNN")
scatter!(p_ssp, Float64.(c_anchor_true), Float64.(SSP_ANCHOR_DEPTHS);
         ms = 4, label = "Five shallow SSP measurements")
save_plot(p_ssp, SSP_PLOT_FILE)
```

## 14. Evaluate the field over the area of interest

Subsample the dense AOI grid onto 0.5 m spacing, predict, and compare against
Kraken side by side (black lines mark the 650–700 m training band):

```julia
df_aoi = CSV.read(joinpath(@__DIR__, AOI_FILE), DataFrame)
#tolerant float comparison, keeps only points sitting on the 0.5 m subgrid
ongrid(x, x0, s) = abs(rem(x - x0, s, RoundNearest)) < 1e-6
df_small = sort!(df_aoi[ongrid.(df_aoi.range_m, 550.0, 0.5) .& ongrid.(df_aoi.depth_m, 1.0, 0.5), :],
                 [:range_m, :depth_m])
ranges = sort(unique(Float64.(df_small.range_m)))
depths = sort(unique(Float64.(df_small.depth_m)))

amp_true = Float64.(df_small.amp)
amp_pred = Float64.(amplitude_output(model, ps, st, make_input(df_small.range_m, df_small.depth_m)))
db_true = 20 .* log10.(amp_true .+ 1e-30)
db_pred = 20 .* log10.(amp_pred .+ 1e-30)
db_err = db_pred .- db_true
println("AOI dB RMSE = ", sqrt(mean(db_err .^ 2)))
CSV.write(joinpath(@__DIR__, AOI_PRED_FILE),
          DataFrame(range_m = Float64.(df_small.range_m), depth_m = Float64.(df_small.depth_m),
                    amp_true = amp_true, amp_pred = amp_pred,
                    db_true = db_true, db_pred = db_pred, db_error = db_err))

nz, nr = length(depths), length(ranges)
hm(v, ttl) = heatmap(ranges, depths, reshape(v, nz, nr); yflip = true, clims = (-85, -45),
                     c = :thermal, title = ttl, xlabel = "Range (m)", ylabel = "Depth (m)",
                     colorbar_title = "dB")
p1 = hm(db_true, "Kraken ground truth")
p2 = hm(db_pred, "Paper-style SSNN-MBNN")
vline!(p1, [650.0, 700.0]; c = :black, lw = 1.2, label = "")
vline!(p2, [650.0, 700.0]; c = :black, lw = 1.2, label = "")
save_plot(plot(p1, p2; layout = (1, 2), size = (1300, 430), dpi = 150, margin = 5 * Plots.mm),
          AOI_PLOT_FILE)
```

## 15. Plot the training history

Train/validation field MSE for the best restart:

```julia
bh = history[history.restart .== best.restart, :]
p_hist = plot(bh.epoch, bh.train_field_mse; yscale = :log10, lw = 2, label = "Training field MSE",
              xlabel = "Epoch", ylabel = "MSE", title = "Best restart training history")
plot!(p_hist, bh.epoch, bh.val_field_mse; lw = 2, label = "Validation field MSE")
save_plot(p_hist, HISTORY_PLOT_FILE)
```

Done — outputs land next to the script: the training history, the named
parameter CSV, the SSP and AOI prediction CSVs, and the three PNGs.
