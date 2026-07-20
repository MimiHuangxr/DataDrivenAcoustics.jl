using DataDrivenAcoustics
using UnderwaterAcoustics
using StableRNGs
using Test
using Statistics
using CSV, DataFrames
using Random

###Ray Models

# prepare dataset
rng = StableRNG(27)
env = UnderwaterEnvironment(seabed=Rock, bathymetry=200.0)
pm = PekerisRayTracer(env; max_bounces=3)
tx = AcousticSource(0, -11, 250)
rxpos = rand(rng, 2, 1000) .* [200.0, 40.0] .+ [5500.0, -110.0]
rxs = [AcousticReceiver(rxpos[1,i], rxpos[2,i]) for i ∈ 1:size(rxpos,2)]
xloss = Float32.(transmission_loss(pm, tx, rxs))

# train data-driven model
pm = DataDrivenPropagationModel(RayBasisNN_2D(60); rng=StableRNG(42))
rxs = [AcousticReceiver(x, z) for (x, z) ∈ zip(rxpos[1,:], rxpos[2,:])]
loss = TransmissionLossMSE(pm, AcousticSource(nothing, 250), rxs, xloss)
DataDrivenAcoustics.fit!(pm, loss; optimizer=Adam(5e-6), minloss=100, maxiters=5000)
DataDrivenAcoustics.fit!(pm, loss; optimizer=BFGS(), maxiters=200)

@test loss(pm.params, nothing) < 5

###Modal Models

using Lux, Zygote, Optimisers

include(joinpath(@__DIR__, "..", "src", "MBNN.jl"))
using .MBNN

# prepare dataset from a known Pekeris environment
rng = StableRNG(1224)
D, freq = 25.0, 500.0
env = UnderwaterEnvironment(bathymetry=D, soundspeed=1500.0, seabed=SandyClay)
pm = PekerisModeSolver(env; nmodes=6)
tx = AcousticSource(0.0, -5.0, freq)
rxpos = rand(rng, 2, 200) .* [50.0, 22.0] .+ [650.0, -23.0]
rxs = [AcousticReceiver(rxpos[1,i], rxpos[2,i]) for i ∈ 1:size(rxpos,2)]
xamp = Float32.(abs.(acoustic_field(pm, tx, rxs)))

# model construction and parameter initialisation
model = ModalBasisNN_2D(D, freq; nmodes=6, nhidden=12, rref=675.0)
ps, st = Lux.setup(rng, model)

@test model.nmodes == 6
@test length(model.ζ) == 201
@test length(ps.A_re) == 6 && length(ps.ssp.W1) == 12

# learned sound speed and wavenumbers respect their physical bounds
c = sound_speed_grid(model, ps)
kr = horizontal_wavenumbers(model, ps)

@test length(c) == 201
@test all(model.cmin .< c .< model.cmax)
@test length(kr) == 6
@test all(model.klo .< kr .< model.khi)
@test issorted(kr; rev=true)

# depth interpolation matrix rows are convex weights
W = depth_interpolation_matrix(model, Float32[0, 1, 2, 3, 4])

@test size(W) == (5, 201)
@test all(≈(1), sum(W; dims=2))
@test all(W .>= 0)

# forward pass returns finite real/imaginary pressure
X = Float32.(vcat(rxpos[1,:]', abs.(rxpos[2,:]')))
yfield, _ = model(X, ps, st)

@test size(yfield) == (2, size(X, 2))
@test all(isfinite, yfield)

# amplitude output is non-negative and consistent with the forward pass
amp = amplitude_output(model, ps, st, X)

@test length(amp) == size(X, 2)
@test all(isfinite, amp)
@test all(amp .>= 0)
@test amp ≈ hypot.(yfield[1,:], yfield[2,:])

# train briefly and check the loss decreases; the loop lives in a function so
# the parameter updates are not lost to soft scope
function train_mbnn(model, ps, st, X, xamp; iters=300)
  yscale = Float32(mean(Float64.(xamp)))
  y_train = xamp ./ yscale
  objective(p) = mean(abs2, amplitude_output(model, p, st, X) ./ yscale .- y_train)
  initial_loss = objective(ps)
  opt_state = Optimisers.setup(
    Optimisers.OptimiserChain(Optimisers.ClipNorm(100.0), Optimisers.Adam(1f-3)), ps)
  for _ ∈ 1:iters
    l, grads = Zygote.withgradient(objective, ps)
    isfinite(l) || error("non-finite loss during MBNN training")
    opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
  end
  ps, initial_loss, objective(ps)
end

ps, initial_loss, final_loss = train_mbnn(model, ps, st, X, xamp)

@test isfinite(initial_loss)
@test isfinite(final_loss)
@test final_loss < initial_loss

# the sound speed profile stays physical after training
c_trained = sound_speed_grid(model, ps)

@test all(model.cmin .< c_trained .< model.cmax)
@test maximum(c_trained) - minimum(c_trained) < 100f0
