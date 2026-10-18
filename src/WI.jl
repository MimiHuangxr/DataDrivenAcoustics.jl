import ChainRulesCore
using SpecialFunctions: besselh

# Receivers are processed in chunks of this size, so the Hankel weights
# (receivers × integration points, complex) stay a few tens of MB even when the
# layer is evaluated on a large prediction grid.
const _WI_CHUNK = 2000

"""
    WavenumberIntegrationNN_2D(D, f, zs; nhidden=8, cmin=1400.0, cmax=1600.0,
                               cinit=1500.0, cbinit=1800.0, ρinit=1.5,
                               attinit=0.1, ngrid=101, rmax=4000.0,
                               cref=soundspeed())

A 2D data-aided wavenumber-integration (learnable fast-field program) layer for
waveguide depth `D` (m), source frequency `f` (Hz) and source depth `zs` (m,
negative below the surface). Calling the layer with a 3×N input matrix of
ranges (row 1), depths (row 2, negative below the surface) and wavenumbers
(row 3) returns a 2×N matrix of real and imaginary acoustic pressure.

The spectral kernel is not learned directly. At every kᵣ on the integration
grid it is the Green's function of the depth-separated equation

    g'' + (k²(z) − kᵣ²) g = −δ(z − zs) / 2π,   k(z) = ω / c(z),

built from a pressure-release surface solution φ₁ and a fluid half-space seabed
solution φ₂ joined at the source by their Wronskian 𝒲,

    g(kᵣ, z) = −φ₁(z<) φ₂(z>) / (2π 𝒲),

and the field is the wavenumber integral along the contour kᵣ − iε,

    p(r, z) = s · ½ Σₘ g(kᵣₘ, z) H₀⁽¹⁾(kᵣₘ r) kᵣₘ Δkᵣ wₘ,

where wₘ is a Hann taper at both ends of the grid and s is a complex source
factor. The trainable parameters are the environment: a sound-speed network
c(z) (the same one-hidden-layer ReLU network as `ModalBasisNN_2D`), the seabed
sound speed, density ratio and attenuation, and s.

Training notes:
- The depth solver mutates arrays, so use forward-mode AD (`AutoForwardDiff()`);
  Zygote is not supported.
- Kernel evaluation is threaded over kᵣ; start Julia with `-t auto`.
- The model assumes e^{-iωt} time dependence (outgoing H₀⁽¹⁾). Data in the
  opposite convention should be conjugated before training.
- The misfit has local minima in the seabed sound speed; scanning `cbinit`
  over a coarse range and keeping the best starting point is recommended.

Fields:
- `nhidden`: hidden width of the sound-speed network (SSNN)
- `D`: total water depth (m)
- `ω`: angular source frequency, ω = 2πf (rad/s)
- `dz`, `zgrid`: depth grid spacing and grid (m, positive downwards)
- `isrc`: index of the source on the depth grid
- `cmin`, `cmax`, `cinit`: water sound-speed bounds and initial value (m/s)
- `cb_lo`, `cb_hi`, `cbinit`: seabed sound-speed bounds and initial value (m/s)
- `ρ_lo`, `ρ_hi`, `ρinit`: seabed/water density ratio bounds and initial value
- `att_hi`, `attinit`: seabed attenuation bound and initial value (dB/λ)
- `kr`, `κ`, `Δkr`, `taper`: integration grid, its contour points kᵣ − iε, spacing
  and Hann taper weights
- `cref`: reference sound speed used to interpret row 3; must match the
  `soundspeed` given to `DataDrivenPropagationModel`

Every term of the sum carries a factor e^{εr} that cancels only in the full
integral, so a sharp edge in the integrand leaves an error growing as e^{εr};
the grid is therefore tapered at both ends (0 → `klo_taper` and ω/cmin → khi).
Computation is in Float64 for the same reason.
"""
struct WavenumberIntegrationNN_2D <: LuxCore.AbstractLuxLayer
  nhidden::Int
  D::Float64
  ω::Float64
  dz::Float64
  zgrid::Vector{Float64}
  isrc::Int
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
  kr::Vector{Float64}
  κ::Vector{ComplexF64}
  Δkr::Float64
  taper::Vector{Float64}
  cref::Float64
end

export WavenumberIntegrationNN_2D, wavenumber_kernel, seabed_parameters

function WavenumberIntegrationNN_2D(D, f, zs; nhidden::Int=8, cmin=1400.0,
                                    cmax=1600.0, cinit=1500.0,
                                    cb_lo=1500.0, cb_hi=2500.0, cbinit=1800.0,
                                    ρ_lo=1.0, ρ_hi=3.0, ρinit=1.5,
                                    att_hi=1.0, attinit=0.1,
                                    ngrid::Int=101, rmax=4000.0, klo_taper=0.02,
                                    cref=soundspeed())
  nhidden > 0 || error("nhidden must be positive")
  ngrid >= 3 || error("ngrid must be at least 3")
  D > 0 || error("D must be positive")
  f > 0 || error("f must be positive")
  rmax > 0 || error("rmax must be positive")
  cref > 0 || error("cref must be positive")
  cmin < cinit < cmax || error("need cmin < cinit < cmax")
  cb_lo < cbinit < cb_hi || error("need cb_lo < cbinit < cb_hi")
  ρ_lo < ρinit < ρ_hi || error("need ρ_lo < ρinit < ρ_hi")
  0 < attinit < att_hi || error("need 0 < attinit < att_hi")
  zs < 0 || error("zs must be negative (below the surface)")
  dz = Float64(D) / (ngrid - 1)
  zgrid = collect(range(0.0, Float64(D); length=ngrid))
  # the source is snapped to the nearest depth node; it must be interior so
  # the Wronskian can be evaluated by central differences
  isrc = round(Int, -zs / dz) + 1
  2 <= isrc <= ngrid - 1 || error("source depth must lie strictly inside (0, D)")
  ω = 2π * Float64(f)
  # Δkᵣ = 2π/(2 rmax) and the usual FFP offset ε = 3/(rmax log₁₀ e); with both
  # ends tapered the sum is converged to ≈ 0.01 dB out to rmax
  klo, khi = 1e-3, 1.2 * ω / cmin
  ε = 3 / (rmax * log10(ℯ))
  nk = ceil(Int, (khi - klo) * rmax / π) + 1
  kr = collect(range(klo, khi; length=nk))
  ktop = ω / cmin
  top = [k <= ktop ? 1.0 : 0.5 * (1 + cos(π * (k - ktop) / (khi - ktop))) for k ∈ kr]
  bottom = [k >= klo_taper ? 1.0 : 0.5 * (1 - cos(π * k / klo_taper)) for k ∈ kr]
  WavenumberIntegrationNN_2D(nhidden, Float64(D), ω, dz, zgrid, isrc,
                             Float64(cmin), Float64(cmax), Float64(cinit),
                             Float64(cb_lo), Float64(cb_hi), Float64(cbinit),
                             Float64(ρ_lo), Float64(ρ_hi), Float64(ρinit),
                             Float64(att_hi), Float64(attinit),
                             kr, kr .- im * ε, kr[2] - kr[1], top .* bottom,
                             Float64(cref))
end

## interface methods

function LuxCore.initialparameters(rng::AbstractRNG, l::WavenumberIntegrationNN_2D)
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
    # complex source factor; 4π converts the −δ/2π source normalisation of the
    # depth equation to a unit-amplitude point source (|p| = 1 at 1 m)
    s_re = 4π,
    s_im = 0.0,
  )
end

LuxCore.initialstates(::AbstractRNG, ::WavenumberIntegrationNN_2D) = NamedTuple()

"""
    sound_speed_grid(l::WavenumberIntegrationNN_2D, ps)

Return the learned water sound-speed profile c(z) on the depth grid, bounded
to (cmin, cmax).
"""
sound_speed_grid(l::WavenumberIntegrationNN_2D, ps) =
  SampledField(_ffp_sound_speed(l, ps); z = .-l.zgrid)

"""
    seabed_parameters(l::WavenumberIntegrationNN_2D, ps)

Return the learned seabed as `(c = …, ρ = …, α = …)`: sound speed (m/s),
seabed/water density ratio and attenuation (dB/λ).
"""
function seabed_parameters(l::WavenumberIntegrationNN_2D, ps)
  (c = l.cb_lo + (l.cb_hi - l.cb_lo) * sigmoid(ps.seabed.qc),
   ρ = l.ρ_lo + (l.ρ_hi - l.ρ_lo) * sigmoid(ps.seabed.qρ),
   α = l.att_hi * sigmoid(ps.seabed.qα))
end

function (l::WavenumberIntegrationNN_2D)(inp::AbstractMatrix, ps, st::NamedTuple)
  size(inp, 1) == 3 || error("input must have exactly 3 rows [x; z; k]; got $(size(inp, 1))")
  r = @view inp[1, :]
  z = @view inp[2, :]
  all(z .<= 0) || error("depths must satisfy z ≤ 0 (negative below the surface)")
  _check_frequency(l, @view inp[3, :])
  # the kernel depends only on the environment, so it is computed once and
  # shared by every chunk of receivers
  G = wavenumber_kernel(l, ps)
  p = reduce(vcat, [_wi_field_chunk(l, G, r[idx], z[idx])
                    for idx ∈ Iterators.partition(eachindex(r), _WI_CHUNK)])
  p = complex(ps.s_re, ps.s_im) .* p
  vcat(reshape(real.(p), 1, :), reshape(imag.(p), 1, :)), st
end

"""
    wavenumber_kernel(l::WavenumberIntegrationNN_2D, ps)

Return the kernel g(kᵣ, z) for the learned environment as an ngrid × nk complex
matrix (rows: depth grid `l.zgrid`; columns: integration grid `l.kr`).
"""
function wavenumber_kernel(l::WavenumberIntegrationNN_2D, ps)
  k2 = (l.ω ./ _ffp_sound_speed(l, ps)) .^ 2
  cb, ρb, att = seabed_parameters(l, ps)
  kb = l.ω / cb * (1 + im * att / (40π * log10(ℯ)))
  T = promote_type(eltype(k2), typeof(cb), typeof(ρb), typeof(att))
  nz, nk, dz, isrc = length(l.zgrid), length(l.kr), l.dz, l.isrc
  G = zeros(Complex{T}, nz, nk)
  h2 = dz^2 / 12
  # every kᵣ is independent, so the contour points are split across threads;
  # each writes only its own column of G
  Threads.@threads for m ∈ 1:nk
    φ1 = zeros(Complex{T}, nz)
    φ2 = zeros(Complex{T}, nz)
    q = k2 .- l.κ[m]^2
    # surface solution: φ₁(0) = 0, φ₁'(0) = 1, integrated downwards by Numerov
    φ1[2] = dz
    for i ∈ 2:nz-1
      φ1[i+1] = (2φ1[i] * (1 - 5h2 * q[i]) - φ1[i-1] * (1 + h2 * q[i-1])) /
                (1 + h2 * q[i+1])
    end
    # seabed solution: decays as e^{−γ(z−D)} below D; continuity of p and
    # (1/ρ) ∂p/∂z gives φ₂'(D) = −γ φ₂(D) / (ρb/ρw); integrated upwards
    γ = sqrt(l.κ[m]^2 - kb^2)
    d = -γ / ρb
    φ2[nz] = 1
    φ2[nz-1] = 1 - dz * d - dz^2 / 2 * q[nz]
    for i ∈ nz-1:-1:2
      φ2[i-1] = (2φ2[i] * (1 - 5h2 * q[i]) - φ2[i+1] * (1 + h2 * q[i+1])) /
                (1 + h2 * q[i-1])
    end
    𝒲 = (φ1[isrc] * (φ2[isrc+1] - φ2[isrc-1]) -
         φ2[isrc] * (φ1[isrc+1] - φ1[isrc-1])) / (2dz)
    c = -1 / (2π * 𝒲)
    for i ∈ 1:nz
      G[i, m] = i <= isrc ? c * φ1[i] * φ2[isrc] : c * φ1[isrc] * φ2[i]
    end
  end
  G
end

## private methods

# tapered Hankel weights and depth stencil depend only on receiver geometry,
# never on a trainable parameter, so they are kept off the AD tape
function _wi_field_chunk(l::WavenumberIntegrationNN_2D, G, r, z)
  Φ, i0, w = ChainRulesCore.ignore_derivatives() do
    rr = max.(Float64.(r), Float64(_RANGE_FLOOR))
    Φ = besselh.(0, 1, rr .* transpose(l.κ)) .* transpose(l.κ .* l.taper) .* (l.Δkr / 2)
    x = clamp.(-Float64.(z) ./ l.dz, 0, length(l.zgrid) - 1)
    i0 = min.(floor.(Int, x), length(l.zgrid) - 2) .+ 1
    Φ, i0, x .- (i0 .- 1)
  end
  Gr = (1 .- w) .* G[i0, :] .+ w .* G[i0 .+ 1, :]
  vec(sum(Φ .* Gr; dims=2))
end

# water sound speed on the depth grid from the SSNN, bounded to (cmin, cmax)
function _ffp_sound_speed(l::WavenumberIntegrationNN_2D, ps)
  zz = 2 .* (l.zgrid ./ l.D) .- 1
  h = max.(zz .* ps.ssp.W1' .+ ps.ssp.b1', 0)
  l.cmin .+ (l.cmax - l.cmin) .* sigmoid.(h * ps.ssp.W2 .+ ps.ssp.b2)
end

# row 3 carries k = ω/cref, built by acoustic_field from the source frequency;
# the kernel is tied to the construction-time frequency, so a mismatch is an
# error rather than something the layer can adapt to
function _check_frequency(l::WavenumberIntegrationNN_2D, k)
  ktarget = l.ω / l.cref
  atol = 1e-3 * ktarget
  if !all(abs.(k .- ktarget) .<= atol)
    j = argmax(abs.(k .- ktarget))
    fq = k[j] * l.cref / 2π
    error("WavenumberIntegrationNN_2D was built for $(l.ω / 2π) Hz " *
          "but queried at $(fq) Hz")
  end
  nothing
end
