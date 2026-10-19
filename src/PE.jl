import ChainRulesCore
using SpecialFunctions: besselh

# dB/λ → imaginary part of k/|k|
const _PE_DBλ = 1 / (40π * log10(ℯ))

"""
    ParabolicEquationNN_2D(D, f, zs; nhidden=8, c0=1500.0, cmin=1400.0,
                           cmax=1600.0, cinit=1500.0, cbinit=1800.0,
                           ρinit=1.5, attinit=0.1, dz=0.5, dr=2.0,
                           zmax=3D, zsponge=2D, sp_att=20.0, npade=4,
                           cref=soundspeed())

A 2D data-aided parabolic-equation layer for waveguide depth `D` (m), source
frequency `f` (Hz) and source depth `zs` (m, negative below the surface).
Calling the layer with a 3×N input matrix of ranges (row 1), depths (row 2,
negative below the surface) and wavenumbers (row 3) returns a 2×N matrix of
real and imaginary acoustic pressure.

The field is written as p(r, z) = s ψ(r, z) H₀⁽¹⁾(k₀r), and the envelope is
marched out in range with Collins' split-step Padé method,

    ∂ψ/∂r = ik₀(√(1 + X) − 1) ψ,   √(1 + X) − 1 ≈ Σⱼ aⱼ X / (1 + bⱼ X),

where X = [ρ ∂z(ρ⁻¹∂z) + k²(z) − k₀²] / k₀² is discretised by finite
differences, with the density at half nodes so the jump at the seabed is
exact. Each Padé term is advanced by a Crank–Nicolson step. The march starts
from Greene's source and ends in an absorbing sponge below the seabed.

The trainable parameters are the environment: a sound-speed network c(z) (the
same one-hidden-layer ReLU network as `ModalBasisNN_2D`), the seabed sound
speed, density ratio and attenuation, and a complex source factor s.

Training notes:
- Gradients through the march come from a hand-written adjoint, registered as
  a ChainRules `rrule`, so `AutoZygote()` gives a gradient for about the cost
  of three marches regardless of the number of parameters. `AutoForwardDiff()`
  also works, but its cost grows with the number of parameters.
- The model assumes e^{-iωt} time dependence (outgoing H₀⁽¹⁾). Data in the
  opposite convention should be conjugated before training.
- The misfit has local minima in the seabed sound speed; scanning `cbinit`
  over a coarse range and keeping the best starting point is recommended.

With the defaults, the march agrees with a converged wavenumber integral to
about 0.4 dB rms from 1 to 4 km in a 200 m waveguide at 100 Hz; `dz = 0.25`
halves the error at twice the cost.

Fields:
- `nhidden`: hidden width of the sound-speed network (SSNN)
- `D`: total water depth (m)
- `ω`, `k0`, `c0`: angular frequency, PE reference wavenumber and sound speed
- `dz`, `dr`: depth and range steps (m)
- `zn`, `iD`: interior depth nodes (m, positive downwards) and the seabed node
- `starter`: Greene's starting field on `zn`
- `sponge`: absorbing-layer attenuation on `zn` (dB/λ)
- `apade`, `bpade`: Padé coefficients
- `cmin`, `cmax`, `cinit`: water sound-speed bounds and initial value (m/s)
- `cb_lo`, `cb_hi`, `cbinit`: seabed sound-speed bounds and initial value (m/s)
- `ρ_lo`, `ρ_hi`, `ρinit`: seabed/water density ratio bounds and initial value
- `att_hi`, `attinit`: seabed attenuation bound and initial value (dB/λ)
- `cref`: reference sound speed used to interpret row 3; must match the
  `soundspeed` given to `DataDrivenPropagationModel`
"""
struct ParabolicEquationNN_2D <: LuxCore.AbstractLuxLayer
  nhidden::Int
  D::Float64
  ω::Float64
  k0::Float64
  c0::Float64
  dz::Float64
  dr::Float64
  zn::Vector{Float64}
  iD::Int
  starter::Vector{Float64}
  sponge::Vector{Float64}
  apade::Vector{Float64}
  bpade::Vector{Float64}
  cmin::Float64
  cmax::Float64
  cinit::Float64
  cb_lo::Float64
  cb_hi::Float64
  cbinit::Float64
  ρ_lo::Float64
  ρ_hi::Float64
  ρinit::Float64
  att_hi::Float64
  attinit::Float64
  cref::Float64
end

export ParabolicEquationNN_2D, seabed_parameters

function ParabolicEquationNN_2D(D, f, zs; nhidden::Int=8, c0=1500.0,
                                cmin=1400.0, cmax=1600.0, cinit=1500.0,
                                cb_lo=1500.0, cb_hi=2500.0, cbinit=1800.0,
                                ρ_lo=1.0, ρ_hi=3.0, ρinit=1.5,
                                att_hi=1.0, attinit=0.1,
                                dz=0.5, dr=2.0, zmax=3D, zsponge=2D, sp_att=20.0,
                                npade::Int=4, cref=soundspeed())
  nhidden > 0 || error("nhidden must be positive")
  npade > 0 || error("npade must be positive")
  D > 0 && f > 0 && c0 > 0 && cref > 0 || error("D, f, c0 and cref must be positive")
  dz > 0 && dr > 0 || error("dz and dr must be positive")
  D < zsponge < zmax || error("need D < zsponge < zmax")
  cmin < cinit < cmax || error("need cmin < cinit < cmax")
  cb_lo < cbinit < cb_hi || error("need cb_lo < cbinit < cb_hi")
  ρ_lo < ρinit < ρ_hi || error("need ρ_lo < ρinit < ρ_hi")
  0 < attinit < att_hi || error("need 0 < attinit < att_hi")
  -D < zs < 0 || error("zs must lie between the surface and the seabed (negative)")
  # the seabed must sit on a node for the density interface to be exact
  isapprox(D / dz, round(D / dz)) || error("D must be a multiple of dz")
  N = round(Int, zmax / dz) - 1                  # interior nodes; ψ = 0 at 0 and zmax
  zn = dz .* collect(1:N)
  iD = round(Int, D / dz)
  ω = 2π * Float64(f)
  k0 = ω / c0
  # Greene's starter with its surface image, so ψ(r, 0) = 0
  A = k0^2 .* (zn .+ zs) .^ 2
  B = k0^2 .* (zn .- zs) .^ 2
  starter = sqrt(k0) .* ((1.4467 .- 0.4201 .* A) .* exp.(-A ./ 3.0512) .-
                         (1.4467 .- 0.4201 .* B) .* exp.(-B ./ 3.0512))
  sponge = sp_att .* clamp.((zn .- zsponge) ./ (zmax - zsponge), 0.0, 1.0) .^ 2
  apade = [2 / (2npade + 1) * sin(j * π / (2npade + 1))^2 for j ∈ 1:npade]
  bpade = [cos(j * π / (2npade + 1))^2 for j ∈ 1:npade]
  ParabolicEquationNN_2D(nhidden, Float64(D), ω, k0, Float64(c0), Float64(dz),
                         Float64(dr), zn, iD, starter, sponge, apade, bpade,
                         Float64(cmin), Float64(cmax), Float64(cinit),
                         Float64(cb_lo), Float64(cb_hi), Float64(cbinit),
                         Float64(ρ_lo), Float64(ρ_hi), Float64(ρinit),
                         Float64(att_hi), Float64(attinit), Float64(cref))
end

## interface methods

function LuxCore.initialparameters(rng::AbstractRNG, l::ParabolicEquationNN_2D)
  bounded(x, lo, hi) = logit(clamp((x - lo) / (hi - lo), _LOGIT_MARGIN, 1 - _LOGIT_MARGIN))
  (
    # one-input, one-hidden-layer ReLU SSNN, as in ModalBasisNN_2D
    ssp = (W1 = 1.5 .* randn(rng, l.nhidden),
           b1 = 0.25 .* randn(rng, l.nhidden),
           W2 = 0.02 .* randn(rng, l.nhidden),
           b2 = bounded(l.cinit, l.cmin, l.cmax)),
    seabed = (qc = bounded(l.cbinit, l.cb_lo, l.cb_hi),
              qρ = bounded(l.ρinit, l.ρ_lo, l.ρ_hi),
              qα = bounded(l.attinit, 0.0, l.att_hi)),
    # complex source factor, absorbing the unknown source level and phase
    s_re = 1.0,
    s_im = 0.0,
  )
end

LuxCore.initialstates(::AbstractRNG, ::ParabolicEquationNN_2D) = NamedTuple()

"""
    sound_speed_grid(l::ParabolicEquationNN_2D, ps)

Return the learned water sound-speed profile c(z) on the water depth nodes,
bounded to (cmin, cmax).
"""
sound_speed_grid(l::ParabolicEquationNN_2D, ps) =
  SampledField(_pe_sound_speed(l, ps); z = .-l.zn[1:l.iD])

"""
    seabed_parameters(l::ParabolicEquationNN_2D, ps)

Return the learned seabed as `(c = …, ρ = …, α = …)`: sound speed (m/s),
seabed/water density ratio and attenuation (dB/λ).
"""
function seabed_parameters(l::ParabolicEquationNN_2D, ps)
  (c = l.cb_lo + (l.cb_hi - l.cb_lo) * sigmoid(ps.seabed.qc),
   ρ = l.ρ_lo + (l.ρ_hi - l.ρ_lo) * sigmoid(ps.seabed.qρ),
   α = l.att_hi * sigmoid(ps.seabed.qα))
end

function (l::ParabolicEquationNN_2D)(inp::AbstractMatrix, ps, st::NamedTuple)
  size(inp, 1) == 3 || error("input must have exactly 3 rows [x; z; k]; got $(size(inp, 1))")
  r = Float64.(inp[1, :])
  z = Float64.(inp[2, :])
  all(z .<= 0) || error("depths must satisfy z ≤ 0 (negative below the surface)")
  all(z .> -l.zn[end]) || error("receivers must lie above the bottom of the domain")
  _check_frequency(l, @view inp[3, :])
  lo, di, up = _pe_operator(l, ps)
  ψ = _pe_march(l, lo, di, up, r, z)
  H = ChainRulesCore.ignore_derivatives() do
    besselh.(0, 1, l.k0 .* max.(r, Float64(_RANGE_FLOOR)))
  end
  p = complex(ps.s_re, ps.s_im) .* ψ .* H
  vcat(reshape(real.(p), 1, :), reshape(imag.(p), 1, :)), st
end

## private methods

# water sound speed on the water nodes from the SSNN, bounded to (cmin, cmax)
function _pe_sound_speed(l::ParabolicEquationNN_2D, ps)
  zz = 2 .* (l.zn[1:l.iD] ./ l.D) .- 1
  h = max.(zz .* ps.ssp.W1' .+ ps.ssp.b1', 0)
  l.cmin .+ (l.cmax - l.cmin) .* sigmoid.(h * ps.ssp.W2 .+ ps.ssp.b2)
end

# Tridiagonal X = (lo, di, up): lo[i] multiplies ψ[i-1], di[i] ψ[i], up[i] ψ[i+1].
# Written without mutation so reverse-mode AD can pass through it. With the
# density at half nodes, ρᵢ/ρ(i−½) = 1 at every node, and ρᵢ/ρ(i+½) = 1 except
# on the seabed node, where it is 1/ρb.
function _pe_operator(l::ParabolicEquationNN_2D, ps)
  N, iD = length(l.zn), l.iD
  cw = _pe_sound_speed(l, ps)
  cb, ρb, att = seabed_parameters(l, ps)
  kw2 = (l.ω ./ cw) .^ 2
  ks2 = (l.ω / cb .* (1 .+ im .* (att .+ l.sponge) .* _PE_DBλ)) .^ 2
  # water nodes, the interface node (averaged), then sediment and sponge
  k2 = vcat(complex.(kw2[1:iD-1]), (kw2[iD] + ks2[iD]) / 2, ks2[iD+1:N])
  s = 1 / (l.dz^2 * l.k0^2)
  lo = fill(s, N)
  up = vcat(fill(s, iD - 1), s / ρb, fill(s, N - iD))
  di = -(lo .+ up) .+ (k2 .- l.k0^2) ./ l.k0^2
  lo, di, up
end

# For each Padé term: A = 1 + cl X (solved), B = 1 + cr X (multiplied), and
# with `adjoint = true` the conjugate transpose Aᴴ as well
function _pe_terms(l::ParabolicEquationNN_2D, lo, di, up; adjoint=false)
  N = length(di)
  σ = l.k0 * l.dr
  map(zip(l.apade, l.bpade)) do (a, b)
    cl, cr = b - im * σ * a / 2, b + im * σ * a / 2
    Al, Ad, Au = cl .* lo, 1 .+ cl .* di, cl .* up
    AH = adjoint ? _trifactor([i == 1 ? zero(Au[1]) : conj(Au[i-1]) for i ∈ 1:N], conj.(Ad),
                              [i == N ? zero(Al[1]) : conj(Al[i+1]) for i ∈ 1:N]) : nothing
    (cl = cl, cr = cr, Bl = cr .* lo, Bd = 1 .+ cr .* di, Bu = cr .* up,
     A = _trifactor(Al, Ad, Au), AH = AH)
  end
end

# march ψ from r = 0 and return it at the receivers (r, z); with keep = true
# every intermediate state is returned too. Each receiver reads ψ by linear
# interpolation between the two range steps and two depth nodes around it.
function _pe_march(l::ParabolicEquationNN_2D, lo, di, up, r, z; keep=false)
  terms = _pe_terms(l, lo, di, up)
  plan = _pe_plan(l, r, z)
  T = eltype(terms[1].Bd)
  ψ = T.(l.starter)
  rhs = similar(ψ)
  out = zeros(T, length(r))
  states = keep ? [copy(ψ)] : nothing
  _pe_record!(out, plan[1], ψ)
  for j ∈ 1:length(plan)-1
    for t ∈ terms
      _trimul!(rhs, t.Bl, t.Bd, t.Bu, ψ)
      _trisolve!(ψ, t.A, rhs)
      keep && push!(states, copy(ψ))
    end
    _pe_record!(out, plan[j + 1], ψ)
  end
  keep ? (out, states, plan) : out
end

# Adjoint of the march: march forward once keeping every state, then march the
# sensitivity λ backwards once through the conjugate-transposed Padé steps,
# collecting the sensitivity to every entry of X. With dL = Re Σ conj(Δ) dψ for
# the output cotangent Δ, the input cotangents follow the ChainRules convention
# (2 ∂L/∂x* for complex x, ∂L/∂x for real x).
function ChainRulesCore.rrule(::typeof(_pe_march), l::ParabolicEquationNN_2D, lo, di, up, r, z)
  out, states, plan = _pe_march(l, lo, di, up, r, z; keep=true)
  function _pe_march_pullback(Δ)
    Δ = ChainRulesCore.unthunk(Δ)
    N = length(di)
    terms = _pe_terms(l, lo, di, up; adjoint=true)
    g = ComplexF64.(Δ) ./ 2                      # ∂L/∂ψ* at the receivers
    Glo, Gdi, Gup = zeros(ComplexF64, N), zeros(ComplexF64, N), zeros(ComplexF64, N)
    λ, μ, v, tmp = zeros(ComplexF64, N), zeros(ComplexF64, N), zeros(ComplexF64, N), zeros(ComplexF64, N)
    _pe_seed!(λ, plan[end], g)
    k = length(states)
    for j ∈ length(plan)-1:-1:1
      for t ∈ reverse(terms)
        ψnew, ψold = states[k], states[k-1]
        k -= 1
        copyto!(tmp, λ)
        _trisolve!(μ, t.AH, tmp)                 # μ = A⁻ᴴ λ
        @. v = t.cr * ψold - t.cl * ψnew
        for i ∈ 1:N
          Gdi[i] += μ[i] * conj(v[i])
          i > 1 && (Glo[i] += μ[i] * conj(v[i-1]))
          i < N && (Gup[i] += μ[i] * conj(v[i+1]))
        end
        for i ∈ 1:N                              # λ = Bᴴ μ
          acc = conj(t.Bd[i]) * μ[i]
          i > 1 && (acc += conj(t.Bu[i-1]) * μ[i-1])
          i < N && (acc += conj(t.Bl[i+1]) * μ[i+1])
          λ[i] = acc
        end
      end
      _pe_seed!(λ, plan[j], g)
    end
    tangent(G, x) = eltype(x) <: Real ? 2 .* real.(G) : 2 .* G
    (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(),
     tangent(Glo, lo), tangent(Gdi, di), tangent(Gup, up),
     ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent())
  end
  out, _pe_march_pullback
end

# plan[j + 1] lists (receiver, range weight, depth node, depth weight) at step j
function _pe_plan(l::ParabolicEquationNN_2D, r, z)
  N = length(l.zn)
  nsteps = ceil(Int, maximum(r) / l.dr) + 1
  plan = [Tuple{Int,Float64,Int,Float64}[] for _ ∈ 0:nsteps]
  for i ∈ eachindex(r)
    j = min(floor(Int, r[i] / l.dr), nsteps - 1)
    t = r[i] / l.dr - j
    x = -z[i] / l.dz
    n = clamp(floor(Int, x), 1, N - 1)
    s = x - n
    push!(plan[j + 1], (i, 1 - t, n, s))
    push!(plan[j + 2], (i, t, n, s))
  end
  plan
end

function _pe_record!(out, entries, ψ)
  for (i, w, n, s) ∈ entries
    out[i] += w * ((1 - s) * ψ[n] + s * ψ[n + 1])
  end
end

# the adjoint of _pe_record!: receiver sensitivities are spread back onto ψ
function _pe_seed!(λ, entries, g)
  for (i, w, n, s) ∈ entries
    λ[n] += w * (1 - s) * g[i]
    λ[n + 1] += w * s * g[i]
  end
end

# tridiagonal LU factors, computed once per matrix and reused at every step
function _trifactor(l, d, u)
  n = length(d)
  cp, iv = similar(d), similar(d)
  iv[1] = 1 / d[1]
  cp[1] = u[1] * iv[1]
  for i ∈ 2:n
    iv[i] = 1 / (d[i] - l[i] * cp[i-1])
    cp[i] = u[i] * iv[i]
  end
  (l = l, cp = cp, inv = iv)
end

# solve with the factors; rhs is overwritten
function _trisolve!(x, F, rhs)
  n = length(x)
  rhs[1] *= F.inv[1]
  for i ∈ 2:n
    rhs[i] = (rhs[i] - F.l[i] * rhs[i-1]) * F.inv[i]
  end
  x[n] = rhs[n]
  for i ∈ n-1:-1:1
    x[i] = rhs[i] - F.cp[i] * x[i+1]
  end
  x
end

function _trimul!(o, l, d, u, v)
  n = length(v)
  for i ∈ 1:n
    acc = d[i] * v[i]
    i > 1 && (acc += l[i] * v[i-1])
    i < n && (acc += u[i] * v[i+1])
    o[i] = acc
  end
  o
end

# row 3 carries k = ω/cref, built by acoustic_field from the source frequency;
# the march is tied to the construction-time frequency, so a mismatch is an
# error rather than something the layer can adapt to
function _check_frequency(l::ParabolicEquationNN_2D, k)
  ktarget = l.ω / l.cref
  atol = 1e-3 * ktarget
  if !all(abs.(k .- ktarget) .<= atol)
    j = argmax(abs.(k .- ktarget))
    fq = k[j] * l.cref / 2π
    error("ParabolicEquationNN_2D was built for $(l.ω / 2π) Hz " *
          "but queried at $(fq) Hz")
  end
  nothing
end
