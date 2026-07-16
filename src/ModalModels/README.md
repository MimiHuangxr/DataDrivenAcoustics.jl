# ModalSolverPaper

`ModalSolverPaper` implements a paper-style **Sound Speed Neural Network (SSNN)** combined with a **Modal Basis Neural Network (MBNN)** for underwater acoustic propagation.

The model learns:

- the depth-dependent sound-speed profile `c(z)`,
- horizontal modal wavenumbers `k_r`,
- complex modal coefficients,
- and the acoustic pressure field as a function of range and depth.

The implementation uses `Lux.jl` for model parameter management and `Zygote.jl` for automatic differentiation.

---

## Main Model

The main exported model is:

```julia
ModalBasisNN_2D
```

It accepts receiver range and depth and returns the real and imaginary components of the acoustic pressure field.

The implementation combines:

1. a neural network that predicts the Sound Speed Profile (SSP),
2. a modal representation of the acoustic field,
3. WKB-style depth-dependent modal functions,
4. cylindrical range spreading,
5. modal phase propagation,
6. and interpolation to arbitrary receiver depths.

---

## Features

- Trainable bounded Sound Speed Profile (SSP)
- Trainable horizontal modal wavenumbers
- Complex trainable modal coefficients
- Pekeris mode-solver warm start
- Linear interpolation to arbitrary receiver depths
- Differentiable acoustic forward model
- Complex pressure and amplitude outputs
- Configurable water depth, frequency, mode count, hidden-layer width, and depth resolution

---

## Dependencies

The module uses:

```julia
using Random
import Lux
import Lux: LuxCore, sigmoid
import LogExpFunctions: logit
import UnderwaterAcoustics as UA
import Zygote
```

Install the external packages with:

```julia
using Pkg

Pkg.add([
    "Lux",
    "LogExpFunctions",
    "UnderwaterAcoustics",
    "Zygote",
])
```

---

## Loading the Module

```julia
include("ModalSolver_paper_ssnn.jl")
using .ModalSolverPaper
using Lux
using Random
```

---

## Constructing the Model

```julia
model = ModalBasisNN_2D(
    25.0,
    500.0;
    nmodes = 6,
    nhidden = 12,
    cmin = 1400.0,
    cmax = 1500.0,
    cinit = 1450.0,
    ngrid = 201,
    rref = 675.0,
)
```

### Constructor Arguments

| Argument | Description |
|---|---|
| `D` | Total water depth in metres |
| `f` | Source frequency in hertz |
| `nmodes` | Maximum number of modal components |
| `nhidden` | Number of hidden neurons in the SSNN |
| `cmin` | Minimum allowed sound speed |
| `cmax` | Maximum allowed sound speed |
| `cinit` | Initial sound-speed estimate |
| `ngrid` | Number of depth-grid points |
| `rref` | Reference range used for modal phase |

The constructor verifies that:

- `D > 0`,
- `f > 0`,
- `nmodes > 0`,
- `nhidden > 0`,
- `ngrid >= 3`,
- and `cmin < cinit < cmax`.

---

## Model Configuration

The model stores the following fixed settings:

| Field | Meaning |
|---|---|
| `nmodes` | Number of trainable modal components |
| `nhidden` | Number of hidden SSNN neurons |
| `D` | Water depth |
| `rref` | Reference range |
| `dz` | Depth-grid spacing |
| `ω` | Angular frequency `2πf` |
| `cmin` | Minimum sound speed |
| `cmax` | Maximum sound speed |
| `cinit` | Initial sound-speed estimate |
| `ζ` | Normalized depth grid |
| `klo` | Lower horizontal-wavenumber bound |
| `khi` | Upper horizontal-wavenumber bound |

The normalized depth coordinate is

```math
ζ = z / D
```

with `ζ` ranging from `0` to `1`.

For the default `ngrid = 201`:

```text
ζ = [0.0, 0.005, 0.01, ..., 1.0]
```

---

## Trainable Parameters

Initialize the model with the Lux interface:

```julia
rng = MersenneTwister(1224)
ps, st = Lux.setup(rng, model)
```

The trainable parameters are:

| Parameter | Meaning |
|---|---|
| `A_re` | Real part of modal coefficient `A` |
| `A_im` | Imaginary part of modal coefficient `A` |
| `B_re` | Real part of modal coefficient `B` |
| `B_im` | Imaginary part of modal coefficient `B` |
| `qkr` | Unconstrained horizontal-wavenumber parameters |
| `ssp.W1` | First-layer SSNN weights |
| `ssp.b1` | First-layer SSNN biases |
| `ssp.W2` | SSNN output weights |
| `ssp.b2` | SSNN output bias |

The model has no changing internal state, so `st` is an empty named tuple.

---

## Sound Speed Neural Network

The sound-speed profile is evaluated with:

```julia
c = sound_speed_grid(model, ps)
```

The SSNN uses normalized depth as input:

```math
z_norm = 2ζ - 1
```

The hidden layer is:

```math
h = ReLU(z_norm W1 + b1)
```

The raw output is:

```math
u = h W2 + b2
```

The bounded sound speed is:

```math
c(z) = cmin + (cmax - cmin) sigmoid(u)
```

Therefore:

```math
cmin <= c(z) <= cmax
```

---

## Acoustic Wavenumber

The local acoustic wavenumber is:

```math
k(z) = ω / c(z)
```

It is evaluated with:

```julia
k = _kgrid(model, ps)
```

---

## Horizontal Modal Wavenumbers

Horizontal modal wavenumbers are evaluated with:

```julia
kr = horizontal_wavenumbers(model, ps)
```

Each unconstrained trainable value is mapped into the allowed interval:

```math
kr_m = klo + (khi - klo) sigmoid(q_m)
```

This keeps every horizontal modal wavenumber within the configured bounds.

---

## Pekeris Warm Start

The function

```julia
_pekeris_kr(model)
```

uses `PekerisModeSolver` from `UnderwaterAcoustics.jl` to estimate initial horizontal modal wavenumbers.

The initialization environment uses:

- constant sound speed `cinit`,
- the configured water depth,
- water density `1000 kg/m³`,
- a fluid seabed density of `2700 kg/m³`,
- and seabed sound speed `5000 m/s`.

This is used only during parameter initialization.

If the Pekeris solver fails, the code keeps the analytic rigid-bottom estimate.

---

## Input Format

The forward model expects a `2 × N` matrix:

```text
row 1: receiver range
row 2: receiver depth
```

Example:

```julia
ranges = Float32[650, 675, 700]
depths = Float32[5, 10, 15]

X = vcat(
    reshape(ranges, 1, :),
    reshape(depths, 1, :),
)
```

The forward model uses the absolute value of depth, so both positive and negative depth conventions are accepted.

---

## Depth Interpolation

Modal functions are first evaluated on the internal depth grid.

To evaluate them at arbitrary receiver depths, use:

```julia
W = depth_interpolation_matrix(
    model,
    Float32[1, 5, 10, 20],
)
```

The interpolation matrix performs linear interpolation between neighboring grid points.

For each requested depth, the corresponding row contains at most two non-zero weights, and the row sums to one.

---

## Vertical Modal Wavenumber

For each mode:

```math
s_m(z) = k(z)^2 - kr_m^2
```

In propagating regions:

```math
kz_m(z) = sqrt(s_m(z))
```

In evanescent regions:

```math
κ_m(z) = sqrt(-s_m(z))
```

The implementation handles both regions separately.

---

## Cumulative Phase

The depth-dependent modal phase is:

```math
φ_m(z) = integral from 0 to z of kz_m(z') dz'
```

It is approximated using cumulative trapezoidal integration:

```julia
_cumtrapz(dz, kz)
```

This provides the phase value at every point on the internal depth grid.

---

## Evanescent Decay

When `k(z)^2 < kr_m^2`, the modal contribution decays exponentially.

The implemented decay factor is:

```math
exp(- integral from 0 to z of κ_m(z') dz')
```

---

## Modal Depth Functions

The modal depth dependence is represented using trainable complex coefficients `A_m` and `B_m`.

The implementation stores the real and imaginary parts separately:

```math
A_m = A_re + i A_im
```

```math
B_m = B_re + i B_im
```

The depth function includes:

- oscillatory cosine and sine terms,
- a WKB-style `1 / sqrt(kz)` factor,
- and exponential attenuation in evanescent regions.

---

## Range Propagation

For each mode, the range-dependent phase is:

```math
ρ_m(r) = kr_m (r - rref)
```

The cylindrical spreading factor is:

```math
1 / sqrt(r kr_m)
```

Very small ranges are clamped internally to avoid division by zero.

---

## Complex Acoustic Pressure

The modal field is represented as:

```math
p(r,z) = sum over m of [ ψ_m(z) / sqrt(r kr_m) ] exp(i kr_m (r-rref))
```

The model returns the real and imaginary pressure components:

```julia
y, st_new = model(X, ps, st)
```

The output shape is:

```text
2 × N
```

where:

```text
row 1: real pressure
row 2: imaginary pressure
```

Example:

```julia
pressure_real = y[1, :]
pressure_imag = y[2, :]
```

---

## Pressure Amplitude

Pressure amplitude is evaluated with:

```julia
amp = amplitude_output(model, ps, st, X)
```

The amplitude is:

```math
|p| = sqrt(p_re^2 + p_im^2)
```

The result is a vector of length `N`.

---

## Exported API

The module exports:

```julia
ModalBasisNN_2D
_kgrid
sound_speed_grid
amplitude_output
depth_interpolation_matrix
horizontal_wavenumbers
_pekeris_kr
```

---

## Minimal Example

```julia
using Random
using Lux

include("ModalSolver_paper_ssnn.jl")
using .ModalSolverPaper

model = ModalBasisNN_2D(
    25.0,
    500.0;
    nmodes = 6,
    nhidden = 12,
    cmin = 1400.0,
    cmax = 1500.0,
    cinit = 1450.0,
    ngrid = 201,
    rref = 675.0,
)

rng = MersenneTwister(1224)
ps, st = Lux.setup(rng, model)

ranges = Float32[650, 675, 700]
depths = Float32[5, 10, 15]

X = vcat(
    reshape(ranges, 1, :),
    reshape(depths, 1, :),
)

complex_pressure, _ = model(X, ps, st)
amplitude = amplitude_output(model, ps, st, X)
sound_speed = sound_speed_grid(model, ps)

println("Complex pressure:")
println(complex_pressure)

println("Pressure amplitude:")
println(amplitude)

println("Predicted SSP:")
println(sound_speed)
```

---

## Training

This module defines the differentiable forward model, but it does not contain a complete optimization loop.

A separate training script should:

1. load acoustic measurements,
2. construct the range-depth input matrix,
3. initialize the model using `Lux.setup`,
4. define the acoustic-field loss,
5. optionally define SSP measurement or physical-prior losses,
6. calculate gradients using Zygote,
7. update parameters using an optimizer,
8. validate the learned field and SSP.

Example:

```julia
using Statistics
using Zygote

loss, grad = Zygote.withgradient(ps) do p
    prediction = amplitude_output(model, p, st, X)
    mean(abs2, prediction .- target)
end
```

---

## Notes

- `nmodes` is the maximum number of trainable modal components. It is not necessarily the exact number of physical propagating modes.
- The Pekeris solver is used only for initialization.
- The seabed parameters in `_pekeris_kr` are placeholders and should be replaced when a different environment is required.
- The SSP is always bounded between `cmin` and `cmax`.
- Receiver depths are evaluated using linear interpolation.
- The interpolation matrix is excluded from automatic differentiation because it depends only on fixed receiver depths and the fixed model grid.
- Very small ranges are clamped internally to avoid singular range spreading.
- Model accuracy depends on training data, initialization, mode count, SSP bounds, and loss-function design.

---

## Suggested File Location

```text
src/ModalModels/ModalSolver_paper_ssnn.jl
```

A related training script may be placed separately:

```text
examples/train_paper_ssnn_mbnn.jl
```

Automated tests may be placed in:

```text
test/runtests.jl
```

---

## Module File

```text
ModalSolver_paper_ssnn.jl
```

This file contains the complete `ModalSolverPaper` module and its differentiable SSNN-MBNN forward model.
