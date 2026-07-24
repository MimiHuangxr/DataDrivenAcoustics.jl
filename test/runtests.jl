using DataDrivenAcoustics
using UnderwaterAcoustics
using StableRNGs
using Test
using Lux
using Zygote, ADTypes

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

@testset "Modal Models" begin

  # prepare dataset from a known Pekeris environment
  rng = StableRNG(1224)
  D, freq = 25.0, 500.0
  env = UnderwaterEnvironment(bathymetry=D, soundspeed=1500.0,
                              seabed=FluidBoundary(1800.0, 1650.0))
  pm1 = PekerisModeSolver(env; nmodes=6)
  tx = AcousticSource(0.0, -5.0, freq)
  rxpos = rand(rng, 2, 200) .* [50.0, 22.0] .+ [650.0, -23.0]
  rxs = [AcousticReceiver(rxpos[1,i], rxpos[2,i]) for i ∈ 1:size(rxpos,2)]
  xamp = Float32.(abs.(acoustic_field(pm1, tx, rxs)))

  # model construction and parameter initialisation
  model = ModalBasisNN_2D(D, freq; nmodes=6, nhidden=12, rref=675.0,
                          cmin=1400.0, cmax=1600.0, cinit=1500.0)
  ps, st = Lux.setup(rng, model)
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

  # the layer takes a 3-row [x; z; k] input, with z <= 0 below the surface
  krow = model.ω / model.cref
  X = Float32.(vcat(rxpos[1,:]', -abs.(rxpos[2,:]'),
                    fill(krow, 1, size(rxpos, 2))))

  # forward pass returns finite real/imaginary pressure
  yfield, _ = model(X, ps, st)
  @test size(yfield) == (2, size(X, 2))
  @test all(isfinite, yfield)

  # the input contract is enforced rather than silently accepted
  @test_throws ErrorException model(X[1:2, :], ps, st)            # missing k row
  @test_throws ErrorException model(vcat(X[1:1,:], abs.(X[2:2,:]), X[3:3,:]), ps, st)  # z > 0
  @test_throws ErrorException model(vcat(X[1:2,:], 0.6f0 .* X[3:3,:]), ps, st)         # wrong frequency

  # the model trains through the package's own fit!
  pm = DataDrivenPropagationModel(model; rng=StableRNG(42))
  loss = FieldAmplitudeMSE(pm, tx, rxs, xamp; sparsity=1f-6)
  initial_loss = loss(pm.params, nothing)
  DataDrivenAcoustics.fit!(pm, loss, AutoZygote(); optimizer=Adam(1f-3), maxiters=200)
  final_loss = loss(pm.params, nothing)
  @test isfinite(initial_loss)
  @test isfinite(final_loss)
  @test final_loss < initial_loss

  # amplitudes come through the standard API, and are non-negative and finite
  amp = abs.(acoustic_field(pm, tx, rxs))
  @test length(amp) == length(rxs)
  @test all(isfinite, amp)
  @test all(amp .>= 0)
  
  # querying a trained model at the wrong frequency is a loud error
  rx = AcousticReceiver(675.0, -12.0)
  @test acoustic_field(pm, tx, rx) isa Complex
  @test_throws ErrorException acoustic_field(pm, AcousticSource(0.0, -5.0, 300.0), rx)

  # the sound speed profile stays physical after training
  c_trained = sound_speed_grid(pm.model, pm.params)
  @test all(model.cmin .< c_trained .< model.cmax)
  @test maximum(c_trained) - minimum(c_trained) < 100f0

end
