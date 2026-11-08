using DataDrivenAcoustics
using UnderwaterAcoustics
using StableRNGs
using Test

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
  xfield = ComplexF32.(acoustic_field(pm1, tx, rxs))   # complex, not abs.() — matches ComplexAmplitudeMSE

  # model construction and parameter initialisation
  model = ModalBasisNN_2D(D, freq; nmodes=6, nhidden=12, rref=675.0)
  pm0 = DataDrivenPropagationModel(model; rng)
  ps, st = pm0.params, NamedTuple()
  @test length(model.ζ) == 201
  @test length(ps.A_re) == 6 && length(ps.ssp.W1) == 12

  # learned sound speed and wavenumbers respect their physical bounds
  ssp = sound_speed_grid(model, ps)
  c = [ssp(z) for z ∈ range(0, -model.D; length=201)]
  kr = [m.kᵣ for m in arrivals(pm0, tx, rxs[1])]
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
  loss = ComplexAmplitudeMSE(pm, tx, rxs, xfield; sparsity=1f-6)
  initial_loss = loss(pm.params, nothing)
  DataDrivenAcoustics.fit!(pm, loss; optimizer=Adam(1f-3), maxiters=200)
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
  ssp_trained = sound_speed_grid(pm.model, pm.params)
  c_trained = [ssp_trained(z) for z ∈ range(0, -model.D; length=201)]
  @test all(model.cmin .< c_trained .< model.cmax)
  @test maximum(c_trained) - minimum(c_trained) < 100f0

end

###Environment Models

# shared dataset: a small Pekeris waveguide, sampled at 50 random positions
D_env, f_env, zs_env = 50.0, 100.0, -10.0
env_env = UnderwaterEnvironment(bathymetry=D_env, soundspeed=1500.0,
                                seabed=FluidBoundary(1800.0, 1650.0))
tx_env = AcousticSource(0.0, zs_env, f_env)
rxpos_env = rand(StableRNG(7), 2, 50) .* [100.0, 40.0] .+ [400.0, -45.0]
rxs_env = [AcousticReceiver(rxpos_env[1,i], rxpos_env[2,i]) for i ∈ 1:size(rxpos_env,2)]
xfield_env = ComplexF64.(acoustic_field(PekerisModeSolver(env_env), tx_env, rxs_env))

# checks shared by both environment layers; zs are the depths of the SSP grid
function test_environment_model(model, zs)
  pm = DataDrivenPropagationModel(model; rng=StableRNG(42))

  # learned sound speed and seabed respect their physical bounds
  ssp = sound_speed_grid(model, pm.params)
  c = [ssp(z) for z ∈ zs]
  @test all(model.cmin .< c .< model.cmax)
  sb = seabed_parameters(model, pm.params)
  @test model.cb_lo < sb.c < model.cb_hi
  @test model.ρ_lo < sb.ρ < model.ρ_hi
  @test 0 < sb.α < model.att_hi

  # forward pass returns a finite field, and the wrong frequency is a loud error
  p = acoustic_field(pm, tx_env, rxs_env)
  @test size(p) == size(rxs_env)
  @test all(isfinite, p)
  @test_throws ErrorException acoustic_field(pm, AcousticSource(0.0, zs_env, 2 * f_env), rxs_env[1])

  # the model trains through the package's own fit!, with its default AD
  loss = ComplexAmplitudeMSE(pm, tx_env, rxs_env, xfield_env)
  initial_loss = loss(pm.params, nothing)
  DataDrivenAcoustics.fit!(pm, loss; optimizer=BFGS(), maxiters=3)
  final_loss = loss(pm.params, nothing)
  @test isfinite(final_loss)
  @test final_loss < initial_loss
end

@testset "Wavenumber integration models" begin
  model = WavenumberIntegrationNN_2D(D_env, f_env, zs_env; rmax=600.0)
  @test_throws UndefKeywordError WavenumberIntegrationNN_2D(D_env, f_env, zs_env)
  test_environment_model(model, .-model.zgrid)
  @test size(wavenumber_kernel(model, DataDrivenPropagationModel(model).params)) ==
        (length(model.zgrid), length(model.kr))
end

@testset "Parabolic equation models" begin
  model = ParabolicEquationNN_2D(D_env, f_env, zs_env)
  test_environment_model(model, .-model.zn[1:model.iD])
end
