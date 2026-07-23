"""
ModalBasisNN_2D(D, f; nmodes=30, nhidden=32, cmin=1400.0, cmax=1500.0,
                    cinit=1450.0, ngrid=201, rref=675.0, cref=soundspeed())

A 2D modal-basis neural network layer for waveguide depth `D` (m) and source
frequency `f` (Hz). Calling the layer with a 3×N input matrix of ranges (row 1),
depths (row 2, negative below the surface) and wavenumbers (row 3) returns a
2×N matrix of real and imaginary acoustic pressure.

Fields:
- `nmodes`: cap on the number of modes the model can use
- `nhidden`: hidden width of the sound-speed network (SSNN)
- `D`: total water depth (m)
- `rref`: reference range (m)
- `dz`: depth grid spacing (m)
- `ω`: angular source frequency, ω = 2πf (rad/s)
- `cmin`, `cmax`: sound-speed bounds (m/s)
- `cinit`: initial sound-speed guess (m/s)
- `ζ`: normalized depth grid, ζ = z/D ∈ [0, 1]
- `klo`, `khi`: horizontal wavenumber bounds (rad/m)
- `cref`: reference sound speed used to interpret row 3; must match the
  `soundspeed` given to `DataDrivenPropagationModel`
"""
struct ModalBasisNN_2D <: LuxCore.AbstractLuxLayer
  nmodes::Int; nhidden::Int
  D::Float32; rref::Float32; dz::Float32; ω::Float32
  cmin::Float32; cmax::Float32; cinit::Float32
  ζ::Vector{Float32}
  klo::Float32; khi::Float32
  cref::Float32
end

function ModalBasisNN_2D(D, f; nmodes::Int=30, nhidden::Int=32, cmin=1400.0,
                         cmax=1500.0, cinit=1450.0, ngrid::Int=201, rref=675.0,
                         cref=soundspeed())
  nmodes > 0 || error("nmodes must be positive")
  nhidden > 0 || error("nhidden must be positive for unknown-SSP training")
  # at least 3 points needed since interpolation uses neighboring depth points
  ngrid >= 3 || error("ngrid must be at least 3")
  D32, f32 = Float32(D), Float32(f)
  D32 > 0 || error("D must be positive")
  f32 > 0 || error("f must be positive")
  cmin32, cmax32, cinit32 = Float32(cmin), Float32(cmax), Float32(cinit)
  cmin32 < cmax32 || error("cmin must be smaller than cmax")
  cmin32 < cinit32 < cmax32 || error("cinit must lie between cmin and cmax")
  ω = 2f0 * Float32(pi) * f32
  ζ = Float32.(range(0f0, 1f0; length=ngrid))
  dz = D32 / Float32(ngrid - 1)
  kmax = ω / cmin32
  klo, khi = 0.02f0 * kmax, 0.999f0 * kmax
  cref32 = Float32(cref)
  cref32 > 0 || error("cref must be positive")
  ModalBasisNN_2D(nmodes, nhidden, D32, Float32(rref), dz, ω,
                  cmin32, cmax32, cinit32, ζ, klo, khi, cref32)
end

## interface methods

function LuxCore.initialparameters(rng::AbstractRNG, l::ModalBasisNN_2D)
  kref = l.ω / l.cinit
  # analytic rigid-bottom guess, kept as fallback
  kr0 = Float32[sqrt(max(kref^2 - ((m - 0.5f0) * Float32(pi) / l.D)^2, l.klo^2))
                for m in 1:l.nmodes]
  # Pekeris warm start overwrites as many modes as the solver finds; it returns
  # only propagating modes, so remaining slots keep the analytic guess
  try
    kr_pek = _pekeris_kr(l)
    n = min(length(kr_pek), l.nmodes)
    kr0[1:n] .= kr_pek[1:n]
  catch err
    @warn "PekerisModeSolver warm start failed; keeping analytic init" exception = err
  end
  t = clamp.((kr0 .- l.klo) ./ (l.khi - l.klo), 1f-4, 1f0 - 1f-4)
  u0 = clamp((l.cinit - l.cmin) / (l.cmax - l.cmin), 1f-4, 1f0 - 1f-4)
  (
    A_re = 1f-2 .* randn(rng, Float32, l.nmodes),
    A_im = 1f-2 .* randn(rng, Float32, l.nmodes),
    B_re = 1f-2 .* randn(rng, Float32, l.nmodes),
    B_im = 1f-2 .* randn(rng, Float32, l.nmodes),
    qkr = logit.(t),
    # one-input, one-hidden-layer ReLU SSNN
    ssp = (W1 = 1.5f0 .* randn(rng, Float32, l.nhidden),
           b1 = 0.25f0 .* randn(rng, Float32, l.nhidden),
           W2 = 0.02f0 .* randn(rng, Float32, l.nhidden),
           b2 = Float32(logit(u0))),
  )
end

LuxCore.initialstates(::AbstractRNG, ::ModalBasisNN_2D) = NamedTuple()

"""
    sound_speed_grid(l::ModalBasisNN_2D, ps)

Return the learned sound-speed profile c(ζ) on the depth grid `l.ζ`, bounded
to (cmin, cmax).
"""
function sound_speed_grid(l::ModalBasisNN_2D, ps)
  z = reshape(2f0 .* l.ζ .- 1f0, :, 1)
  W1, b1 = reshape(ps.ssp.W1, 1, :), reshape(ps.ssp.b1, 1, :)
  h = max.(z .* W1 .+ b1, 0f0)
  raw = vec(h * ps.ssp.W2 .+ ps.ssp.b2)
  l.cmin .+ (l.cmax - l.cmin) .* sigmoid.(raw)
end

"""
    horizontal_wavenumbers(l::ModalBasisNN_2D, ps)

Return the learned horizontal wavenumbers kᵣ per mode, bounded to (klo, khi).
"""
horizontal_wavenumbers(l::ModalBasisNN_2D, ps) =
  l.klo .+ (l.khi - l.klo) .* sigmoid.(ps.qkr)

function (l::ModalBasisNN_2D)(inp::AbstractMatrix, ps, st::NamedTuple)
  size(inp, 1) == 3 || error("input must have exactly 3 rows [x; z; k]; got $(size(inp, 1))")
  r = @view inp[1, :]
  z = @view inp[2, :]
  all(z .<= 0f0) || error("depths must satisfy z ≤ 0 (negative below the surface)")
  d = -z
  _check_frequency(l, @view inp[3, :])
  r_safe = max.(r, 1f-3)
  kr = horizontal_wavenumbers(l, ps)
  k = _kgrid(l, ps)
  s = k .^ 2 .- reshape(kr .^ 2, 1, :)
  kz = sqrt.(max.(s, 0f0) .+ 1f-8)
  phase_z = _cumtrapz(l.dz, kz)
  κ = sqrt.(max.(-s, 0f0) .+ 1f-8)
  decay_z = exp.(-_cumtrapz(l.dz, κ))
  invsqrt_kz = decay_z ./ sqrt.(kz .+ 1f-6)
  A_re, A_im = reshape(ps.A_re, 1, :), reshape(ps.A_im, 1, :)
  B_re, B_im = reshape(ps.B_re, 1, :), reshape(ps.B_im, 1, :)
  cosφ, sinφ = cos.(phase_z), sin.(phase_z)
  ψre = invsqrt_kz .* ((A_re .+ B_re) .* cosφ .+ (B_im .- A_im) .* sinφ)
  ψim = invsqrt_kz .* ((A_im .+ B_im) .* cosφ .+ (A_re .- B_re) .* sinφ)
  Wdepth = ChainRulesCore.ignore_derivatives() do
    depth_interpolation_matrix(l, d)
  end
  Dre, Dim = Wdepth * ψre, Wdepth * ψim
  range_phase = (r_safe .- l.rref) .* reshape(kr, 1, :)
  range_scale = 1f0 ./ sqrt.(r_safe .* reshape(kr, 1, :))
  cosρ, sinρ = cos.(range_phase), sin.(range_phase)
  yre = sum((Dre .* cosρ .- Dim .* sinρ) .* range_scale; dims=2)
  yim = sum((Dre .* sinρ .+ Dim .* cosρ) .* range_scale; dims=2)
  vcat(reshape(yre, 1, :), reshape(yim, 1, :)), st
end

"""
    amplitude_output(l::ModalBasisNN_2D, ps, st, inp)

Run the forward pass and return the pressure amplitude |p| at each input
(range, depth) column of `inp`.
"""
function amplitude_output(l::ModalBasisNN_2D, ps, st, inp)
  y, _ = l(inp, ps, st)
  hypot.(y[1, :], y[2, :])
end

"""
    depth_interpolation_matrix(l::ModalBasisNN_2D, depths)

Return a length(depths) × ngrid linear interpolation matrix that evaluates
functions on the model depth grid at arbitrary depths (e.g. SSP anchors).
"""
function depth_interpolation_matrix(l::ModalBasisNN_2D, depths)
  d = Float32.(collect(depths))
  n = length(l.ζ)
  W = zeros(Float32, length(d), n)
  for (row, z) in enumerate(d)
    x = clamp(z / l.dz, 0f0, Float32(n - 1))
    i0 = min(floor(Int, x), n - 2)
    w = x - Float32(i0)
    W[row, i0 + 1] = 1f0 - w
    W[row, i0 + 2] = w
  end
  W
end

## private methods

# cumulative trapezoidal integration along dims=1
_cumtrapz(dz::Float32, y::AbstractMatrix) =
  dz .* (cumsum(y; dims=1) .- 0.5f0 .* y .- 0.5f0 .* y[1:1, :])

# wavenumber grid k(z) = ω / c(z) from the learned SSP
_kgrid(l::ModalBasisNN_2D, ps) = l.ω ./ sound_speed_grid(l, ps)

# warm-start kr; only called at initialization, the SSNN takes over afterwards
function _pekeris_kr(l::ModalBasisNN_2D)
  f = Float64(l.ω) / (2π)
  env = UnderwaterEnvironment(
    bathymetry = Float64(l.D),
    soundspeed = Float64(l.cinit),
    density = 1000.0,
    seabed = FluidBoundary(2700.0, 5000.0),  # hard rock-like halfspace; swap for actual seabed
  )
  pm = PekerisModeSolver(env; nmodes=l.nmodes)
  tx = AcousticSource(0.0, -Float64(l.D) / 2, f)
  rx = AcousticReceiver(Float64(l.rref), -Float64(l.D) / 2)
  # kr depends only on the environment, not on tx/rx positions
  modes = arrivals(pm, tx, rx)
  Float32[clamp(Float32(real(m.kᵣ)), l.klo, l.khi) for m in modes]
end

function _check_frequency(l::ModalBasisNN_2D, k)
  ktarget = l.ω / l.cref
  atol = 1f-3 * ktarget
  if !all(abs.(k .- ktarget) .<= atol)
    j = argmax(abs.(k .- ktarget))
    fq = Float32(k[j]) * l.cref / (2f0 * Float32(pi))
    error("ModalBasisNN_2D was built for $(l.ω / (2f0 * Float32(pi))) Hz " *
          "but queried at $(fq) Hz")
  end
  nothing
end
