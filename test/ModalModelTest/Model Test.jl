module PaperSSNNTraining

using CSV, DataFrames, Random, Statistics, LinearAlgebra, Plots
using Lux, Zygote, Optimisers, ProgressMeter
using UnderwaterAcoustics, AcousticsToolbox

#the mbnn + ssnn model lives in its own module
include(joinpath(@__DIR__, "ModalSolver_paper_ssnn.jl"))
using .ModalSolverPaper

gr() #plots backend

# ============================================================
# FILES
# ============================================================
#input
const REFERENCE_SSP_FILE = "reference_ssp_100pts.csv"
#kraken-generated data
const MEASUREMENT_FILE   = "ssnn_profiles_1224.csv"
const AOI_FILE           = "ssnn_ground_truth_aoi.csv"
#training outputs
const HISTORY_FILE       = "paper_ssnn_training_history.csv"
const PARAMETER_FILE     = "paper_ssnn_parameters.csv"
const SSP_PRED_FILE      = "paper_ssnn_predicted_ssp.csv"
const AOI_PRED_FILE      = "paper_ssnn_aoi_prediction.csv"
#plots
const SSP_PLOT_FILE      = "paper_ssnn_true_vs_predicted_ssp.png"
const AOI_PLOT_FILE      = "paper_ssnn_true_vs_predicted_aoi.png"
const HISTORY_PLOT_FILE  = "paper_ssnn_training_history.png"

# Regenerate once so the Kraken data definitely matches this setup.
#change to false after the first successful run to avoid recomputing it
const FORCE_REGENERATE_GROUND_TRUTH = true

# ============================================================
# USER INPUT SETTINGS
# ============================================================
const D = 25.0f0;       const FREQ = 500.0f0;   const SOURCE_DEPTH = 5.0
const NMODES = 6;       const NHIDDEN = 12;     const NGRID = 201
#sound speed bounds plus the initial guess so the model doesnt go haywire
const CMIN = 1400.0f0;  const CMAX = 1500.0f0;  const CINIT = 1450.0f0
const MAX_SSP_SPAN = 35.0f0
const RREF = 675.0 #reference range inside the measurement band
#random restarts escape bad initializations; early stopping checks every LOG_EVERY epochs
#early stopping happens when lowest point achieved (continuous rise after)
const RESTARTS = 10;    const EPOCHS = 8000;    const BATCH_SIZE = 256
const LOG_EVERY = 200;  const EARLY_STOPPING_CHECKS = 12
const LEARNING_RATE = 1.0f-3;  const GRAD_CLIP = 100.0 #clip is a safety valve, rarely triggers at 100

# eqn 22 loss function: field error + L1(A) + L1(B)
const ALPHA = 1.0f-6;  const BETA = 1.0f-6

# Section III-B / Fig. 10: shallow SSP measurements at 0, 1, 2, 3, and 4 m.
const SSP_ANCHOR_DEPTHS = Float32[0, 1, 2, 3, 4]

# the SSP error is normalized by 35 m/s before applying this weight, hence chosen 20
const LAMBDA_SSP = 20.0f0
# Implements the paper's prior that total SSP variation should not exceed 35 m/s.
const LAMBDA_SPAN = 1.0f0

# ============================================================
# HELPERS
# ============================================================
save_plot(plt, f) = (display(plt); savefig(plt, joinpath(@__DIR__, f)); println("Saved: ", joinpath(@__DIR__, f)))
flat(x)          = Optimisers.destructure(x)[1]   # numeric array leaves of any param/grad tree
allfinite(x)     = all(isfinite, flat(x))         #replaces tree_allfinite
make_input(r, d)   = Float32.(vcat(r', d'))       # 2 x N (range; depth)
l1_complex(re, im) = sum(sqrt.(re .^ 2 .+ im .^ 2 .+ 1f-12)) #smoothed so the gradient exists at 0

function load_reference_ssp()
    df = sort!(CSV.read(joinpath(@__DIR__, REFERENCE_SSP_FILE), DataFrame), :depth_m)
    z, c = Float64.(df.depth_m), Float64.(df.c_ms)
    # SampledField used purely as a 1-D linear interpolator on (positive) depth,
    #replaces the old interp1_linear
    return z, c, SampledField(c; z = z, interp = Linear())
end

# ============================================================
# KRAKEN SYNTHETIC DATA
# Matches the paper's 25 m, 500 Hz, 51 x 24 measurement setup.
# ============================================================
#one kraken run on a receiver grid
function sample_field(pm, tx, ranges, depths)
    #acoustic_field on AcousticReceiverGrid2D(ranges, -depths) comes back (nranges x ndepths),
    #the assert catches it if a package update ever changes that
    amp = abs.(acoustic_field(pm, tx, AcousticReceiverGrid2D(ranges, -depths)))
    @assert size(amp) == (length(ranges), length(depths)) "Unexpected acoustic field size: $(size(amp))"
    a = vec(permutedims(amp))   # depth fastest, then range
    DataFrame(
        range_m = repeat(collect(ranges); inner = length(depths)),
        depth_m = repeat(collect(depths); outer = length(ranges)),
        amp = a,
        db = 20 .* log10.(a .+ 1e-30), #tiny offset avoids log10(0)
    )
end

function generate_ground_truth()
    Random.seed!() #fresh entropy, so the train/validation split differs run to run
    z_ref, c_ref, _ = load_reference_ssp()
    println("Generating Kraken data...")
    println("Reference SSP: $(c_ref[1]) m/s at surface -> $(c_ref[end]) m/s at bottom")

    env = UnderwaterEnvironment(
        bathymetry = Float64(D),
        soundspeed = SampledField(c_ref;
            z = range(0.0, -Float64(D); length = length(c_ref)), interp = CubicSpline()),
        seabed = SandyClay,
    )
    pm = Kraken(env; rmax = 850.0, nmodes = NMODES)
    tx = AcousticSource(0.0, 0.0, -SOURCE_DEPTH, Float64(FREQ); spl = 0.0)

    println("Computing AOI ground truth...")
    #dense 0.1 m grid over the whole area of interest, only used for evaluation later
    CSV.write(joinpath(@__DIR__, AOI_FILE), sample_field(pm, tx, 550.0:0.1:800.0, 1.0:0.1:24.0))

    println("Computing 1,224 measurement points...")
    #51 ranges x 24 depths = 1224 synthetic measurements inside the 650-700 m band
    meas = sample_field(pm, tx, 650.0:1.0:700.0, 1.0:1.0:24.0)
    split = fill("validation", nrow(meas))
    split[randperm(nrow(meas))[1:round(Int, 0.70 * nrow(meas))]] .= "train" #70/30 split
    meas.split = split
    CSV.write(joinpath(@__DIR__, MEASUREMENT_FILE), meas)
    println("Saved ", AOI_FILE, " and ", MEASUREMENT_FILE)
end

function ensure_ground_truth()
    have = isfile(joinpath(@__DIR__, MEASUREMENT_FILE)) && isfile(joinpath(@__DIR__, AOI_FILE))
    if FORCE_REGENERATE_GROUND_TRUTH || !have
        generate_ground_truth()
    else
        println("Using existing Kraken CSV files.")
        println("Set FORCE_REGENERATE_GROUND_TRUTH=true if they were generated with another setup.")
    end
end

# ============================================================
# TRAINING
# ============================================================
function main()
    ensure_ground_truth()
    z_ref, c_ref, ref_ssp = load_reference_ssp()

    #train/validation split comes from the csv itself
    df = CSV.read(joinpath(@__DIR__, MEASUREMENT_FILE), DataFrame)
    s = lowercase.(String.(df.split))
    train_df = df[s .== "train", :]
    val_df   = df[(s .== "validation") .| (s .== "val"), :]
    isempty(train_df) && error("No training rows found")
    isempty(val_df)   && error("No validation rows found")

    X_train = make_input(train_df.range_m, train_df.depth_m)
    X_val   = make_input(val_df.range_m, val_df.depth_m)
    yscale  = mean(Float64.(train_df.amp))
    yscale > 0 || error("Mean training amplitude must be positive")
    y_train = Float32.(train_df.amp ./ yscale)
    y_val   = Float32.(val_df.amp ./ yscale)

    model = ModalBasisNN_2D(D, FREQ; nmodes = NMODES, nhidden = NHIDDEN,
                            cmin = CMIN, cmax = CMAX, cinit = CINIT, ngrid = NGRID, rref = RREF)

    # Five shallow CTD measurements, as in Section III-B / Fig. 10.
    c_anchor_true = Float32.(ref_ssp.(Float64.(SSP_ANCHOR_DEPTHS)))
    #fixed interpolation matrix that reads the model's c(z) at the anchor depths
    W_anchor = depth_interpolation_matrix(model, SSP_ANCHOR_DEPTHS)

    println("="^60)
    println("PAPER-STYLE SSNN + MBNN TRAINING")
    println("modes = $NMODES | SSNN hidden units = $NHIDDEN | restarts = $RESTARTS")
    println("training points = $(length(y_train)) | validation points = $(length(y_val))")
    println("SSP anchors = ", collect(zip(SSP_ANCHOR_DEPTHS, c_anchor_true)))
    println("No automatic loss balancing; no smoothness, monotonicity, or surface-boundary penalty")
    println("="^60)

    #eqn 22 pieces plus the two SSP terms, computed once and reused for the logging too
    function components(ps, st, X, y)
        amp = amplitude_output(model, ps, st, X) ./ Float32(yscale)
        cgrid = sound_speed_grid(model, ps)
        span_excess = max(maximum(cgrid) - minimum(cgrid) - MAX_SSP_SPAN, 0f0)
        return (
            field = mean(abs2, amp .- y),
            A = l1_complex(ps.A_re, ps.A_im),
            B = l1_complex(ps.B_re, ps.B_im),
            ssp = mean(abs2, (W_anchor * cgrid .- c_anchor_true) ./ MAX_SSP_SPAN),
            span = (span_excess / MAX_SSP_SPAN)^2,
        )
    end
    total(c) = c.field + ALPHA * c.A + BETA * c.B + LAMBDA_SSP * c.ssp + LAMBDA_SPAN * c.span
    objective(ps, st, X, y) = total(components(ps, st, X, y))

    best = (val = Inf, ps = nothing, st = nothing, restart = 0, epoch = 0)
    history = DataFrame(restart = Int[], epoch = Int[], train_objective = Float64[],
                        train_field_mse = Float64[], val_field_mse = Float64[],
                        anchor_rmse_ms = Float64[], ssp_min_ms = Float64[], ssp_max_ms = Float64[],
                        grad_norm = Float64[])
    ntrain = length(y_train)

    for restart in 1:RESTARTS
        rng = MersenneTwister()
        ps, st = Lux.setup(rng, model)
        opt_state = Optimisers.setup(OptimiserChain(ClipNorm(GRAD_CLIP), Adam(LEARNING_RATE)), ps)

        rbest_val, rbest_ps, rbest_epoch, stale = Inf, deepcopy(ps), 0, 0
        pbar = Progress(EPOCHS; desc = "Restart $restart/$RESTARTS ", barlen = 28, showspeed = true)

        for epoch in 1:EPOCHS
            #fresh random minibatch every epoch
            idxs = rand(rng, 1:ntrain, min(BATCH_SIZE, ntrain))
            loss, grads = Zygote.withgradient(p -> objective(p, st, X_train[:, idxs], y_train[idxs]), ps)
            g = grads[1]
            #stop the restart instead of letting NaNs poison Adam's state
            if !isfinite(loss) || !allfinite(g)
                println("\nRestart $restart stopped: NaN/Inf at epoch $epoch"); break
            end
            opt_state, ps = Optimisers.update(opt_state, ps, g)
            if !allfinite(ps)
                println("\nRestart $restart stopped: invalid parameters"); break
            end

            #full-dataset metrics every LOG_EVERY epochs (plus first and last)
            if epoch == 1 || epoch % LOG_EVERY == 0 || epoch == EPOCHS
                train_c = components(ps, st, X_train, y_train)
                val_c   = components(ps, st, X_val, y_val)
                cgrid = Float64.(sound_speed_grid(model, ps))
                anchor_rmse = sqrt(mean((Float64.(W_anchor * Float32.(cgrid)) .- Float64.(c_anchor_true)) .^ 2))
                val_metric = Float64(val_c.field)
                push!(history, (restart, epoch, Float64(total(train_c)), Float64(train_c.field),
                                val_metric, anchor_rmse, minimum(cgrid), maximum(cgrid),
                                Float64(norm(flat(g)))))
                #early stopping tracks the validation field mse only
                if val_metric < rbest_val - 1e-10
                    rbest_val, rbest_ps, rbest_epoch, stale = val_metric, deepcopy(ps), epoch, 0
                else
                    stale += 1
                end
                next!(pbar; showvalues = [(:epoch, epoch),
                    (:train_field, round(Float64(train_c.field); sigdigits = 5)),
                    (:val_field, round(val_metric; sigdigits = 5)),
                    (:anchor_rmse, round(anchor_rmse; sigdigits = 5)),
                    (:ssp_span, round(maximum(cgrid) - minimum(cgrid); digits = 3))])
                if stale >= EARLY_STOPPING_CHECKS
                    println("\nRestart $restart early-stopped at epoch $epoch; best epoch = $rbest_epoch"); break
                end
            else
                next!(pbar)
            end
        end

        println("Restart $restart best validation field MSE = $rbest_val at epoch $rbest_epoch")
        #keep the best restart overall
        if rbest_val < best.val
            best = (val = rbest_val, ps = rbest_ps, st = st, restart = restart, epoch = rbest_epoch)
        end
    end

    best.ps === nothing && error("All restarts failed")
    ps, st = best.ps, best.st
    println("\n", "="^60)
    println("BEST RUN\nrestart = $(best.restart) | epoch = $(best.epoch) | validation field MSE = $(best.val)")
    println("="^60)
    CSV.write(joinpath(@__DIR__, HISTORY_FILE), history)

    # ========================================================
    # SAVE PARAMETERS
    # every trained parameter flattened into one named csv
    # ========================================================
    pname(s, n) = ["$(s)_$i" for i in 1:n]
    par_names = vcat([pname(s, NMODES) for s in ("A_re", "A_im", "B_re", "B_im", "qkr")]...,
                     [pname(s, NHIDDEN) for s in ("ssp_W1", "ssp_b1", "ssp_W2")]..., "ssp_b2")
    par_vals = Float64.(vcat(ps.A_re, ps.A_im, ps.B_re, ps.B_im, ps.qkr,
                             ps.ssp.W1, ps.ssp.b1, ps.ssp.W2, ps.ssp.b2))
    CSV.write(joinpath(@__DIR__, PARAMETER_FILE), DataFrame(parameter = par_names, value = par_vals))

    # ========================================================
    # SSP EVALUATION (reference queried via the SampledField interpolator)
    # ========================================================
    z_pred = Float64.(model.ζ .* model.D) #model depth grid in metres
    c_pred = Float64.(sound_speed_grid(model, ps))
    c_true = Float64.(ref_ssp.(z_pred))
    err = c_pred .- c_true
    println("SSP RMSE          = ", sqrt(mean(err .^ 2)), " m/s")
    println("SSP max abs error = ", maximum(abs.(err)), " m/s")
    println("Predicted span    = ", maximum(c_pred) - minimum(c_pred), " m/s")
    CSV.write(joinpath(@__DIR__, SSP_PRED_FILE),
              DataFrame(depth_m = z_pred, c_true_ms = c_true, c_pred_ms = c_pred, error_ms = err))

    p_ssp = plot(c_ref, z_ref; yflip = true, lw = 2.5, label = "Reference SSP",
                 xlabel = "Sound speed (m/s)", ylabel = "Depth (m)",
                 title = "Paper-style SSNN inversion",
                 xlims = (1400, 1500), xticks = 1400:25:1500, ylims = (0, 25), yticks = 0:5:25)
    plot!(p_ssp, c_pred, z_pred; lw = 2.5, ls = :dash, label = "Learned SSNN")
    scatter!(p_ssp, Float64.(c_anchor_true), Float64.(SSP_ANCHOR_DEPTHS);
             ms = 4, label = "Five shallow SSP measurements")
    save_plot(p_ssp, SSP_PLOT_FILE)

    # ========================================================
    # AOI EVALUATION (subsample the 0.1 m AOI grid onto a 0.5 m grid)
    # ========================================================
    df_aoi = CSV.read(joinpath(@__DIR__, AOI_FILE), DataFrame)
    #tolerant float comparison, keeps only points sitting on the 0.5 m subgrid
    ongrid(x, x0, s) = abs(rem(x - x0, s, RoundNearest)) < 1e-6
    df_small = sort!(df_aoi[ongrid.(df_aoi.range_m, 550.0, 0.5) .& ongrid.(df_aoi.depth_m, 1.0, 0.5), :],
                     [:range_m, :depth_m])
    ranges = sort(unique(Float64.(df_small.range_m)))
    depths = sort(unique(Float64.(df_small.depth_m)))

    amp_true = Float64.(df_small.amp)
    amp_pred = Float64.(amplitude_output(model, ps, st, make_input(df_small.range_m, df_small.depth_m)))
    db_true = 20 .* log10.(amp_true .+ 1e-30)
    db_pred = 20 .* log10.(amp_pred .+ 1e-30)
    db_err = db_pred .- db_true
    println("AOI amplitude RMSE = ", sqrt(mean((amp_pred .- amp_true) .^ 2)))
    println("AOI dB RMSE        = ", sqrt(mean(db_err .^ 2)))
    CSV.write(joinpath(@__DIR__, AOI_PRED_FILE),
              DataFrame(range_m = Float64.(df_small.range_m), depth_m = Float64.(df_small.depth_m),
                        amp_true = amp_true, amp_pred = amp_pred,
                        db_true = db_true, db_pred = db_pred, db_error = db_err))

    nz, nr = length(depths), length(ranges)
    hm(v, ttl) = heatmap(ranges, depths, reshape(v, nz, nr); yflip = true, clims = (-85, -45),
                         c = :thermal, title = ttl, xlabel = "Range (m)", ylabel = "Depth (m)",
                         colorbar_title = "dB")
    p1 = hm(db_true, "Kraken ground truth")
    p2 = hm(db_pred, "Paper-style SSNN-MBNN")
    #black lines mark the 650-700 m band the model was trained on
    vline!(p1, [650.0, 700.0]; c = :black, lw = 1.2, label = "")
    vline!(p2, [650.0, 700.0]; c = :black, lw = 1.2, label = "")
    save_plot(plot(p1, p2; layout = (1, 2), size = (1300, 430), dpi = 150, margin = 5 * Plots.mm),
              AOI_PLOT_FILE)

    # ========================================================
    # HISTORY PLOT
    # ========================================================
    bh = history[history.restart .== best.restart, :]
    p_hist = plot(bh.epoch, bh.train_field_mse; yscale = :log10, lw = 2, label = "Training field MSE",
                  xlabel = "Epoch", ylabel = "MSE", title = "Best restart training history")
    plot!(p_hist, bh.epoch, bh.val_field_mse; lw = 2, label = "Validation field MSE")
    save_plot(p_hist, HISTORY_PLOT_FILE)

    println("\nSaved:")
    foreach(f -> println("- ", f), (HISTORY_FILE, PARAMETER_FILE, SSP_PRED_FILE, AOI_PRED_FILE,
                                    SSP_PLOT_FILE, AOI_PLOT_FILE, HISTORY_PLOT_FILE))
end

main()

end
