module ModalSolverPaper

# ============================================================
# 1. IMPORTS AND EXPORTS
# ============================================================

using Random
import Lux
import Lux: LuxCore, sigmoid
import LogExpFunctions: logit
import UnderwaterAcoustics as UA
import Zygote

export ModalBasisNN_2D
export _kgrid, sound_speed_grid, amplitude_output
export depth_interpolation_matrix, horizontal_wavenumbers
export _pekeris_kr

# ============================================================
# 2. MODEL CONFIGURATION STRUCTURE
# Stores fixed physical and numerical settings
# ============================================================

struct ModalBasisNN_2D <: LuxCore.AbstractLuxLayer
    nmodes::Int
    nhidden::Int
    D::Float32
    rref::Float32
    dz::Float32
    ω::Float32
    cmin::Float32
    cmax::Float32
    cinit::Float32
    ζ::Vector{Float32} #ζ = [0.0, 0.005, 0.01, ..., 1.0], ζ=z/D​, normalized depth coordinate
    klo::Float32
    khi::Float32
end

# ============================================================
# 3. MATHEMATICAL HELPER FUNCTIONS
# sigmoid comes from Lux/NNlib, logit from LogExpFunctions.
# ============================================================

#trapezoidal integrand, ensures accuracy
_cumtrapz(dz::Float32, y::AbstractMatrix) =
    dz .* (cumsum(y; dims = 1) .- 0.5f0 .* y .- 0.5f0 .* y[1:1, :])

# ============================================================
# 4. MODEL CONSTRUCTOR
# Builds depth grid and wavenumber bounds
# ============================================================

function ModalBasisNN_2D(
    D, #total depth
    f; #source emitter frequency
    #have to be defined here for user input
    nmodes::Int = 30,
    nhidden::Int = 32, 
    #cap of number of modes the model can use
    cmin = 1400.0,
    cmax = 1500.0,
    cinit = 1450.0,
    #provide initial guess such that the model doesnt go haywire
    ngrid::Int = 201,
    rref = 675.0,
)
    nmodes > 0 || error("nmodes must be positive")
    nhidden > 0 || error("nhidden must be positive for unknown-SSP training")
    ngrid >= 3 || error("ngrid must be at least 3")
    #At least 3 points are needed because the code uses neighboring depth points
    #for interpolation and integration. User can pass a number that's not 201 (default).

    D32 = Float32(D)
    f32 = Float32(f)
    D32 > 0 || error("D must be positive")
    f32 > 0 || error("f must be positive")

    cmin32 = Float32(cmin)
    cmax32 = Float32(cmax)
    cinit32 = Float32(cinit)

    cmin32 < cmax32 || error("cmin must be smaller than cmax")
    cmin32 < cinit32 < cmax32 || error("cinit must lie between cmin and cmax")

    ω = 2f0 * Float32(pi) * f32 #ω=2πf
    ζ = Float32.(range(0f0, 1f0; length = ngrid))
    dz = D32 / Float32(ngrid - 1)

    kmax = ω / cmin32
    klo = 0.02f0 * kmax
    khi = 0.999f0 * kmax

    return ModalBasisNN_2D(
        nmodes,
        nhidden,
        D32,
        Float32(rref),
        dz,
        ω,
        cmin32,
        cmax32,
        cinit32,
        ζ,
        klo,
        khi,
    )
end

# ============================================================
# 5. TRAINABLE PARAMETER INITIALIZATION
# Initializes modal coefficients, kr values and SSNN parameters
# ============================================================

# Warm-start kr only called at initialization,
#executes SSNN later.
function _pekeris_kr(l::ModalBasisNN_2D)
    f = Float64(l.ω) / (2π)
    env = UA.UnderwaterEnvironment(
        bathymetry = Float64(l.D),
        soundspeed = Float64(l.cinit),
        density = 1000.0,
        seabed = UA.FluidBoundary(2700.0, 5000.0), #hard rock-like halfspace; user to swap
    )
    pm = UA.PekerisModeSolver(env; nmodes = l.nmodes)
    tx = UA.AcousticSource(0.0, -Float64(l.D) / 2, f)
    rx = UA.AcousticReceiver(Float64(l.rref), -Float64(l.D) / 2)
    modes = UA.arrivals(pm, tx, rx)
    #kr does not depend on tx/rx positions, only on the environment
    return Float32[clamp(Float32(real(m.kᵣ)), l.klo, l.khi) for m in modes]
end

function LuxCore.initialparameters(rng::AbstractRNG, l::ModalBasisNN_2D)
    kref = l.ω / l.cinit

    #creates an initial horizontal wavenumber for every mode
    #(analytic rigid-bottom guess, kept as fallback)
    kr0 = Float32[
        sqrt(max(
            kref^2 - ((m - 0.5f0) * Float32(pi) / l.D)^2,
            l.klo^2,
        ))
        for m in 1:l.nmodes
    ]

    # Pekeris warm start: overwrite as many modes as the solver finds
    # (it returns only propagating modes, so at low f·D there may be
    # fewer than nmodes; remaining slots keep the analytic guess).
    try
        kr_pek = _pekeris_kr(l)
        n = min(length(kr_pek), l.nmodes)
        kr0[1:n] .= kr_pek[1:n]
    catch err
        @warn "PekerisModeSolver warm start failed; keeping analytic init" exception = err
    end

    t = clamp.(
        (kr0 .- l.klo) ./ (l.khi - l.klo),
        1f-4,
        1f0 - 1f-4,
    )

    u0 = clamp(
        (l.cinit - l.cmin) / (l.cmax - l.cmin),
        1f-4,
        1f0 - 1f-4,
    )

    return (
        #creates one coefficient for each mode
        A_re = 1f-2 .* randn(rng, Float32, l.nmodes),
        A_im = 1f-2 .* randn(rng, Float32, l.nmodes),
        B_re = 1f-2 .* randn(rng, Float32, l.nmodes),
        B_im = 1f-2 .* randn(rng, Float32, l.nmodes),
        qkr = logit.(t),
        ssp = (
            # One-input, one-hidden-layer ReLU SSNN.
            W1 = 1.5f0 .* randn(rng, Float32, l.nhidden),
            b1 = 0.25f0 .* randn(rng, Float32, l.nhidden),
            W2 = 0.02f0 .* randn(rng, Float32, l.nhidden),
            b2 = Float32(logit(u0)),
        ),
    )
end

LuxCore.initialstates(::AbstractRNG, ::ModalBasisNN_2D) = NamedTuple()

# ============================================================
# 6. ENVIRONMENT AND WAVENUMBER FUNCTIONS
# SSNN predicts c(z), then converts it into k(z)
# ============================================================

function sound_speed_grid(l::ModalBasisNN_2D, ps)
    z = reshape(2f0 .* l.ζ .- 1f0, :, 1)
    W1 = reshape(ps.ssp.W1, 1, :)
    b1 = reshape(ps.ssp.b1, 1, :)

    h = max.(z .* W1 .+ b1, 0f0)
    raw = vec(h * ps.ssp.W2 .+ ps.ssp.b2)

    return l.cmin .+
           (l.cmax - l.cmin) .* sigmoid.(raw)
end

_kgrid(l::ModalBasisNN_2D, ps) = l.ω ./ sound_speed_grid(l, ps)

function horizontal_wavenumbers(l::ModalBasisNN_2D, ps)
    return l.klo .+
           (l.khi - l.klo) .* sigmoid.(ps.qkr)
end

# ============================================================
# 7. DEPTH INTERPOLATION
# Evaluates modal functions at arbitrary receiver depths
# ============================================================

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

    return W
end

# ============================================================
# 8. MBNN FORWARD PROPAGATION
# Converts range and depth into complex acoustic pressure
# ============================================================

function (l::ModalBasisNN_2D)(inp::AbstractMatrix, ps, st::NamedTuple)
    size(inp, 1) >= 2 || error("input must have at least two rows: range and depth")

    r = @view inp[1, :]
    d = abs.(@view inp[2, :])
    r_safe = max.(r, 1f-3)

    kr = horizontal_wavenumbers(l, ps)
    k = _kgrid(l, ps)

    s = k .^ 2 .- reshape(kr .^ 2, 1, :)

    kz = sqrt.(max.(s, 0f0) .+ 1f-8)
    phase_z = _cumtrapz(l.dz, kz)

    κ = sqrt.(max.(-s, 0f0) .+ 1f-8)
    decay_z = exp.(-_cumtrapz(l.dz, κ))

    invsqrt_kz = decay_z ./ sqrt.(kz .+ 1f-6)

    A_re = reshape(ps.A_re, 1, :)
    A_im = reshape(ps.A_im, 1, :)
    B_re = reshape(ps.B_re, 1, :)
    B_im = reshape(ps.B_im, 1, :)

    cosφ = cos.(phase_z)
    sinφ = sin.(phase_z)

    ψre = invsqrt_kz .* (
        (A_re .+ B_re) .* cosφ .+
        (B_im .- A_im) .* sinφ
    )

    ψim = invsqrt_kz .* (
        (A_im .+ B_im) .* cosφ .+
        (A_re .- B_re) .* sinφ
    )

    Wdepth = Zygote.ignore() do
        depth_interpolation_matrix(l, d)
    end

    Dre = Wdepth * ψre
    Dim = Wdepth * ψim

    range_phase = (r_safe .- l.rref) .* reshape(kr, 1, :)
    range_scale = 1f0 ./ sqrt.(r_safe .* reshape(kr, 1, :))

    cosρ = cos.(range_phase)
    sinρ = sin.(range_phase)

    yre = sum(
        (Dre .* cosρ .- Dim .* sinρ) .* range_scale;
        dims = 2,
    )

    yim = sum(
        (Dre .* sinρ .+ Dim .* cosρ) .* range_scale;
        dims = 2,
    )

    y = vcat(
        reshape(yre, 1, :),
        reshape(yim, 1, :),
    )

    return y, st
end

# ============================================================
# 9. OUTPUT POST-PROCESSING
# Converts complex pressure into pressure amplitude
# ============================================================

function amplitude_output(l::ModalBasisNN_2D, ps, st, inp)
    y, _ = l(inp, ps, st)
    return hypot.(y[1, :], y[2, :])
end

end
