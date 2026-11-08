using UnderwaterAcoustics
import UnderwaterAcoustics: AbstractPropagationModel, AbstractAcousticSource, AbstractAcousticReceiver, ModeArrival
import ComponentArrays: ComponentArray
import Optimization: OptimizationFunction, OptimizationProblem, solve, AutoReverseDiff, AutoForwardDiff
import OptimizationOptimisers: Adam
import OptimizationOptimJL: BFGS, LBFGS
# AD backends for fit!: ReverseDiff by default, ForwardDiff for the
# environment layers (WavenumberIntegrationNN_2D, ParabolicEquationNN_2D)
import ReverseDiff
import ForwardDiff
import SpecialFunctions: besselh
import Random
import ChainRulesCore
import Lux
import Lux: LuxCore, sigmoid

export DataDrivenPropagationModel, TransmissionLossMSE, ComplexAmplitudeMSE
export Adam, BFGS, LBFGS
public fit!

struct DataDrivenPropagationModel{T1,T2} <: AbstractPropagationModel
  model::T1
  params::T2
  c::Float32
end

Base.show(io::IO, m::DataDrivenPropagationModel) = print(io, "DataDrivenPropagationModel(", m.model, ")")

"""
    DataDrivenPropagationModel(model; rng=Random.GLOBAL_RNG, soundspeed=soundspeed())

Data-driven acoustic propagation model using a Lux neural network layer `model`.
The model is initialized with random parameters using `rng` and assumes an isovelocity
medium with sound speed `soundspeed`.
"""
function DataDrivenPropagationModel(model; rng=Random.GLOBAL_RNG, soundspeed=soundspeed())
  pst = Lux.setup(rng, model)
  DataDrivenPropagationModel(model, ComponentArray(pst[1]), Float32(soundspeed))
end

# callable version to create a copy with new parameters
(pm::DataDrivenPropagationModel)(params) = DataDrivenPropagationModel(pm.model, params, pm.c)

function UnderwaterAcoustics.acoustic_field(pm::DataDrivenPropagationModel, tx::AbstractAcousticSource, rx::AbstractAcousticReceiver)
  f = frequency(tx)
  p = location(rx)
  k = 2f0 * π * f / pm.c
  # receiver positions are data, not parameters, so the input carries no gradient
  inp = ChainRulesCore.ignore_derivatives(() -> Float32[p.x; p.z; k;;])
  out = Lux.LuxCore.stateless_apply(pm.model, inp, pm.params)
  complex(out[1], out[2])
end

function UnderwaterAcoustics.acoustic_field(pm::DataDrivenPropagationModel, tx::AbstractAcousticSource, rxs::AbstractArray{<:AbstractAcousticReceiver})
  f = frequency(tx)
  p = vec(location.(rxs))
  k = 2f0 * π * f / pm.c
  # receiver positions are data, not parameters, so the input carries no gradient
  inp = ChainRulesCore.ignore_derivatives(() -> Float32[getfield.(p, :x) getfield.(p, :z) fill(k, length(p))])
  out = Lux.LuxCore.stateless_apply(pm.model, inp', pm.params)
  reshape(complex.(out[1,:], out[2,:]), size(rxs))
end

"""
    arrivals(pm::DataDrivenPropagationModel, tx::AbstractAcousticSource, rx::AbstractAcousticReceiver)

Return the learned modal arrivals for `pm` (wrapping a [`ModalBasisNN_2D`](@ref))
as a vector of `ModeArrival`s. Only `m` (mode number), `kᵣ` (horizontal
wavenumber), and `vₚ` (phase velocity, derived as `ω/kᵣ`) are populated —
`ψ` (mode function) and `v` (group velocity) are `missing`, since this model
has no explicit depth eigenfunction and no dispersion curve to differentiate
for group velocity.
"""
function UnderwaterAcoustics.arrivals(pm::DataDrivenPropagationModel, tx::AbstractAcousticSource, rx::AbstractAcousticReceiver)
  l, ps = pm.model, pm.params
  kr = l.klo .+ (l.khi - l.klo) .* sigmoid.(ps.qkr)
  [ModeArrival(m, kr[m], missing, missing, l.ω / kr[m]) for m in eachindex(kr)]
end

# AD backend that fit! uses for a model unless one is passed explicitly; layers
# whose forward pass updates arrays in place override this
_default_adtype(model) = AutoReverseDiff(compile=true)

function fit!(pm::DataDrivenPropagationModel, loss, adtype=_default_adtype(pm.model); optimizer=Adam(1e-4), maxiters=100, minloss=0, show_progress=0)
  ofun = OptimizationFunction(loss, adtype)
  oprob = OptimizationProblem(ofun, pm.params)
  cb = let show_progress = show_progress, minloss = minloss
    (st, l) -> begin
      show_progress > 0 && st.iter % show_progress == 0 && println("Iteration $(st.iter), Loss: $l")
      l < minloss
    end
  end
  sol = solve(oprob, optimizer; maxiters, callback=cb)
  pm.params .= sol.u
  pm
end

"""
    TransmissionLossMSE(pm, tx, rxs, data; sparsity=10f0)

Loss function for fitting a data-driven propagation model `pm` to observed
transmission loss data `data` measured at receivers `rxs`. The frequency of
operation is determined by the source `tx`. The `sparsity` parameter controls
the L1 regularization strength on the ray amplitudes to promote sparsity in
the solution.
"""
function TransmissionLossMSE(pm::DataDrivenPropagationModel, tx::AbstractAcousticSource, rxs::AbstractArray{<:AbstractAcousticReceiver}, data; sparsity=10f0)
  let tx = tx, rxs = rxs, data = data, sparsity = sparsity
    (ps, _) -> sum(abs2, transmission_loss(pm(ps), tx, rxs) - data) + sparsity * sum(abs, ps.A)
  end
end

"""
    ComplexAmplitudeMSE(pm, tx, rxs, data; sparsity=1f-6)

Loss function for fitting a data-driven propagation model `pm` to observed
complex field data `data` (magnitude and phase, e.g. from a full solver such as
Kraken or Pekeris) measured at receivers `rxs`. Unlike
[`TransmissionLossMSE`](@ref), which fits amplitude only, this loss fits the
complex field directly, so it makes use of phase information when it is
available. The frequency of operation is determined by the source `tx`. For
models with modal amplitude coefficients (e.g. [`ModalBasisNN_2D`](@ref)), the
`sparsity` parameter controls the L1 regularization strength on those
coefficients; models without them are not regularized.
"""
function ComplexAmplitudeMSE(pm::DataDrivenPropagationModel, tx::AbstractAcousticSource, rxs::AbstractArray{<:AbstractAcousticReceiver}, data; sparsity=1f-6)
  let pm = pm, tx = tx, rxs = rxs, data = data, sparsity = sparsity
    (ps, _) -> sum(abs2, vec(acoustic_field(pm(ps), tx, rxs)) - data) +
               sparsity * _sparsity_penalty(pm.model, ps)
  end
end

# L1 regularization used by ComplexAmplitudeMSE; models without sparse
# coefficients add nothing, and models that have them define their own method
_sparsity_penalty(model, ps) = 0

## helpers shared by the physics-based layers

# Smallest range used when evaluating a field; avoids division by zero at r = 0.
const _RANGE_FLOOR = 1f-3

# Keeps sigmoid-bounded parameters away from the interval endpoints, where their
# logit is infinite.
const _LOGIT_MARGIN = 1f-4

logit(p) = log(p / (1 - p))

# unconstrained parameter that sigmoid maps back to x in (lo, hi)
_bounded_logit(x, lo, hi) = logit(clamp((x - lo) / (hi - lo), _LOGIT_MARGIN, 1 - _LOGIT_MARGIN))

# one-input, one-hidden-layer ReLU sound-speed network (SSNN) evaluated at
# normalised depths x ∈ [-1, 1], bounded to (cmin, cmax)
function _ssnn_sound_speed(ssp, x, cmin, cmax)
  h = max.(x .* reshape(ssp.W1, 1, :) .+ reshape(ssp.b1, 1, :), 0)
  cmin .+ (cmax - cmin) .* sigmoid.(h * ssp.W2 .+ ssp.b2)
end

# row 3 carries k = ω/cref, built by acoustic_field from the source frequency;
# each layer is tied to its construction-time frequency, so a mismatch is an
# error rather than something the layer can adapt to
function _check_frequency(l, k)
  ktarget = l.ω / l.cref
  atol = 1f-3 * ktarget
  if !all(abs.(k .- ktarget) .<= atol)
    j = argmax(abs.(k .- ktarget))
    error("$(nameof(typeof(l))) was built for $(l.ω / 2π) Hz " *
          "but queried at $(k[j] * l.cref / 2π) Hz")
  end
  nothing
end

# Layers that learn a water sound-speed profile and a fluid seabed. Their
# forward passes update arrays in place, so fit! trains them with forward-mode AD.
abstract type _EnvironmentLayer <: LuxCore.AbstractLuxLayer end

_default_adtype(::_EnvironmentLayer) = AutoForwardDiff()

"""
    seabed_parameters(l, ps)

Return the seabed learned by a [`WavenumberIntegrationNN_2D`](@ref) or
[`ParabolicEquationNN_2D`](@ref) layer `l` with parameters `ps`, as
`(c = …, ρ = …, α = …)`: sound speed (m/s), seabed/water density ratio and
attenuation (dB/λ).
"""
function seabed_parameters(l::_EnvironmentLayer, ps)
  (c = l.cb_lo + (l.cb_hi - l.cb_lo) * sigmoid(ps.seabed.qc),
   ρ = l.ρ_lo + (l.ρ_hi - l.ρ_lo) * sigmoid(ps.seabed.qρ),
   α = l.att_hi * sigmoid(ps.seabed.qα))
end

export seabed_parameters
