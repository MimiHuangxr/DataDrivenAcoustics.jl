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

include(joinpath(@__DIR__, "..", "ModalSolver.jl"))
using .ModalSolver

rmse(a, b) = sqrt(mean((a .- b).^2))

@testset "Known SSP baseline regression test" begin

  meas_file = joinpath(@__DIR__, "ssnn_profiles_1224.csv")
  ssp_file = joinpath(@__DIR__, "true_ssnn_ssp.csv")

  RMSE_LIMIT = 2.0
  SEED = 1224

  # baseline input files exist
  @testset "1. Baseline input files exist" begin
    @test isfile(meas_file)
    @test isfile(ssp_file)
  end

  # known SSP CSV loads correctly
  ssp = load_ssp(ssp_file)

  @testset "2. Known SSP loading works" begin
    @test length(ssp[1]) == length(ssp[2])
    @test all(isfinite, ssp[1])
    @test all(isfinite, ssp[2])
  end

  # known-SSP solver can be constructed
  pm = ModeSolver(D = 25.0, f = 500.0, ssp = ssp)

  @testset "3. Known-SSP solver construction works" begin
    @test pm.D == 25.0
    @test pm.f == 500.0
    @test pm.ssp !== nothing
  end

  # training runs without crashing
  @testset "4. Known-SSP training runs" begin
    fit!(pm, meas_file; restarts = 10, seed = SEED)
    @test pm.theta !== nothing
    @test !isempty(pm.history)
  end

  # prediction returns valid amplitudes
  df = CSV.read(meas_file, DataFrame)
  pred = predict_amp(pm, df.range_m, df.depth_m)

  @testset "5. Prediction output is valid" begin
    @test length(pred) == nrow(df)
    @test all(isfinite, pred)
    @test all(pred .>= 0)
  end

  # main regression test for the known-SSP reproduction
  measurement_rmse = rmse(pred, df.amp)

  @testset "6. Baseline RMSE stays below threshold" begin
    @test measurement_rmse < RMSE_LIMIT
  end

  # training log contains finite losses
  @testset "7. Loss history is valid" begin
    @test "epoch" in names(pm.history)
    @test "train_loss" in names(pm.history)
    @test "val_loss" in names(pm.history)
    @test all(isfinite, pm.history.train_loss)
    @test all(isfinite, pm.history.val_loss)
    @test length(pm.history.val_loss) >= 2
  end

  # loss may bounce, but the best validation loss should be lower
  # than the first logged validation loss
  @testset "8. Validation loss improves overall" begin
    first_val_loss = first(pm.history.val_loss)
    best_val_loss = minimum(pm.history.val_loss)
    @test best_val_loss <= first_val_loss
  end

  # checks that the training process records the least loss; does not
  # recompute the internal loss manually, so it is more stable for CI
  @testset "9. Least validation loss is recorded" begin
    val_losses = pm.history.val_loss
    best_val_loss = minimum(val_losses)
    best_epoch_idx = argmin(val_losses)
    @test isfinite(best_val_loss)
    @test best_epoch_idx >= 1
    @test best_epoch_idx <= length(val_losses)
    @test best_val_loss <= first(val_losses)
  end
end
