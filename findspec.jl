using OrdinaryDiffEq
using Roots
using Plots
using LaTeXStrings
using Printf

# -----------------------------------------------------------------------
# Fixed parameter
# -----------------------------------------------------------------------
const c_fixed = -1.0

# -----------------------------------------------------------------------
# ODE: -U'' + (ax + c + γ²)U = 0,  U(0)=0, U'(0)=1
# -----------------------------------------------------------------------
function resonance_ode!(du, u, p, x)
    a, c, γ2 = p
    du[1] = u[2]
    du[2] = (a*x + c + γ2) * u[1]
end

function shoot(γ::Real, a::Real; c::Real=c_fixed)
    p    = (Float64(a), Float64(c), Float64(γ)^2)
    u0   = [0.0, 1.0]
    prob = ODEProblem(resonance_ode!, u0, (0.0, 1.0), p)
    return solve(prob, Tsit5(); abstol=1e-14, reltol=1e-14, save_everystep=true)
end

resonance_condition_scalar(γ, a) = begin
    sol = shoot(γ, a)
    U1, dU1 = sol[end]
    dU1 - γ*U1
end

# -----------------------------------------------------------------------
# Trapezoidal integration
# -----------------------------------------------------------------------
function trapz(xs, ys)
    s = 0.0
    @inbounds for i in 1:length(xs)-1
        s += (xs[i+1] - xs[i]) * (ys[i] + ys[i+1]) / 2
    end
    return s
end

function diagnostics(γ::Real, a::Real; c::Real=c_fixed)
    sol  = shoot(γ, a; c=c)
    ts   = sol.t
    Us   = [sol(x)[1] for x in ts]
    int4 = trapz(ts, Us .^ 4)
    int2 = trapz(ts, Us .^ 2)
    U1   = sol(1.0)[1]

    numer = U1^4 - 4γ * int4
    denom = 2*U1^2 - 4γ * int2
    Edot  = -numer / denom
    Ω     = U1^4 / (2γ) - 2*int4
    pos   = all(u > -1e-10 for u in Us[2:end-1])

    return (Edot=Edot, Ω=Ω, U1=U1, int4=int4, int2=int2, pos=pos)
end

# -----------------------------------------------------------------------
# Seed: scan γ for sign changes at fixed a
# -----------------------------------------------------------------------
function seed_resonances(a_seed; γ_min=0.01, γ_max=50.0, n_scan=4000)
    γs = range(γ_min, γ_max; length=n_scan)
    Fs = [resonance_condition_scalar(γ, a_seed) for γ in γs]
    seeds = Float64[]
    for i in 1:length(γs)-1
        if isfinite(Fs[i]) && isfinite(Fs[i+1]) && Fs[i]*Fs[i+1] < 0
            γ0 = find_zero(γ -> resonance_condition_scalar(γ, a_seed),
                           (γs[i], γs[i+1]), Bisection(); atol=1e-13, rtol=1e-13)
            push!(seeds, γ0)
        end
    end
    return sort(seeds)
end

# -----------------------------------------------------------------------
# Direct-scan continuation: track γ(a) by Newton at each a step
# -----------------------------------------------------------------------
function continue_branch(γ_seed::Float64, a_seed::Float64;
                         a_end::Float64=2.0,
                         n_steps::Int=1000)
    av = collect(range(a_seed, a_end; length=n_steps))
    γv = fill(NaN, n_steps)
    γv[1] = γ_seed

    for i in 2:n_steps
        γ_prev = γv[i-1]
        isnan(γ_prev) && continue
        γv[i] = try
            find_zero(γ -> resonance_condition_scalar(γ, av[i]),
                      γ_prev, Order1(); atol=1e-13)
        catch
            NaN
        end
    end
    return av, γv
end

# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

a_start = 1.0
a_end   = 2.0

# 1. Seed all branches at a = a_start
println("Seeding resonances at a = $a_start, c = $c_fixed ...")
γ_seeds = seed_resonances(a_start)
println("  Found $(length(γ_seeds)) branch seeds: γ = $γ_seeds")

# 2. Continue each branch from a_start to a_end
branches = []
for (k, γ0) in enumerate(γ_seeds)
    println("Continuing branch $k  (γ₀ = $(round(γ0,digits=5)), a₀ = $a_start) ...")
    av, γv = continue_branch(γ0, a_start; a_end=a_end)
    push!(branches, (a=av, γ=γv))
    n_ok = count(!isnan, γv)
    println("  Branch $k done — $n_ok valid points")
end

# 3. Evaluate diagnostics along each branch
branch_diag = []
for (k, br) in enumerate(branches)
    av   = br.a
    γv   = br.γ
    Ev   = Float64[]
    Ωv   = Float64[]
    posv = Bool[]
    av_ok = Float64[]
    γv_ok = Float64[]

    for i in eachindex(av)
        isnan(γv[i]) && continue
        d = diagnostics(γv[i], av[i])
        push!(av_ok, av[i])
        push!(γv_ok, γv[i])
        push!(Ev,    d.Edot)
        push!(Ωv,    d.Ω)
        push!(posv,  d.pos)
    end

    push!(branch_diag, (a=av_ok, γ=γv_ok, Edot=Ev, Ω=Ωv, pos=posv))
end

# 4. Find zeros of Ė
println("\n===== Zeros of Ė =====")
zero_crossings = []

for (k, bd) in enumerate(branch_diag)
    Ev = bd.Edot; av = bd.a; γv = bd.γ
    for i in 1:length(Ev)-1
        (isnan(Ev[i]) || isnan(Ev[i+1])) && continue
        if Ev[i] * Ev[i+1] < 0
            max(abs(Ev[i]), abs(Ev[i+1])) > 10.0 && continue

            t           = Ev[i] / (Ev[i] - Ev[i+1])
            a_star      = av[i] + t * (av[i+1] - av[i])
            γ_star_init = γv[i] + t * (γv[i+1] - γv[i])

            Δγ     = abs(γv[i+1] - γv[i]) + 1e-6
            γ_lo_b = max(1e-8, γ_star_init - Δγ)
            γ_hi_b = γ_star_init + Δγ
            γ_star = try
                fa_b = resonance_condition_scalar(γ_lo_b, a_star)
                fb_b = resonance_condition_scalar(γ_hi_b, a_star)
                if fa_b * fb_b < 0
                    find_zero(γ -> resonance_condition_scalar(γ, a_star),
                              (γ_lo_b, γ_hi_b), Bisection(); atol=1e-13)
                else
                    γ_star_init
                end
            catch
                γ_star_init
            end
            d = diagnostics(γ_star, a_star)

            push!(zero_crossings, (branch=k, a_star=a_star, γ_star=γ_star,
                                   Edot=d.Edot, Ω=d.Ω, U1=d.U1, pos=d.pos))

            @printf("Branch %d:  a★ = %.14f\n", k, a_star)
            @printf("           γ★ = %.14f\n", γ_star)
            @printf("           Ė  = %.6e  (should be ≈ 0)\n", d.Edot)
            @printf("           Ω  = %.10f\n", d.Ω)
            @printf("           U(1) = %.10f\n", d.U1)
            @printf("           U★ > 0 on (0,1): %s\n\n", d.pos)
        end
    end
end

if isempty(zero_crossings)
    println("No zeros of Ė detected on any branch in a ∈ [$a_start, $a_end].")
end

# -----------------------------------------------------------------------
# 5. Plots
# -----------------------------------------------------------------------
colors = [:blue, :red, :green4, :purple, :darkorange, :brown]

pγ = plot(xlabel="a", ylabel="γ",  legend=:topright, size=(700,420))
pE = plot(xlabel="a", ylabel="Ė",  legend=:topright, size=(700,420))
pΩ = plot(xlabel="a", ylabel="Ω",  legend=:topright, size=(700,420))

for (k, bd) in enumerate(branch_diag)
    col = colors[mod1(k, length(colors))]
    lbl = "branch $k"
    plot!(pγ, bd.a, bd.γ;    label=lbl, color=col, lw=2)
    plot!(pE, bd.a, bd.Edot; label=lbl, color=col, lw=2)
    plot!(pΩ, bd.a, bd.Ω;    label=lbl, color=col, lw=2)
end

for zc in zero_crossings
    col = colors[mod1(zc.branch, length(colors))]
    scatter!(pE, [zc.a_star], [0.0];        markersize=10, color=col, label="Ė=0 branch $(zc.branch)")
    scatter!(pγ, [zc.a_star], [zc.γ_star]; markersize=10, color=col, label="")
    scatter!(pΩ, [zc.a_star], [zc.Ω];      markersize=10, color=col, label="")
end

hline!(pE, [0.0]; linestyle=:dash, color=:black, label="Ė=0", lw=1.5)
hline!(pΩ, [0.0]; linestyle=:dash, color=:black, label="Ω=0", lw=1.5)

fig = plot(pγ, pE, pΩ; layout=(3,1), size=(800,1100), left_margin=5Plots.mm)
display(fig)
savefig(fig, "resonance_branches.png")
println("\nFigure saved to resonance_branches.png")

# -----------------------------------------------------------------------
# 6. V and U_★ at each Ė = 0 point
# -----------------------------------------------------------------------
for zc in zero_crossings
    a_s = zc.a_star
    γ_s = zc.γ_star

    xs_V = range(0.0, 1.0; length=500)
    Vs   = @. a_s * xs_V + c_fixed

    pV = plot(xs_V, Vs;
        xlabel = L"x", ylabel = L"V(x)",
        label = false, lw = 2.5, color = :red,
        size = (500, 300))
    hline!(pV, [0.0]; linestyle=:dot, color=:black, lw=1, label="")

    sol_u = shoot(γ_s, a_s)
    xu = sol_u.t
    Uu = [u[1] for u in sol_u.u]

    pU = plot(xu, Uu;
        xlabel = L"x", ylabel = L"U_{\star}(x)",
        label = false, lw = 2.5,
        xlims = (0, 1),
        ylims = (0, max(maximum(Uu), 0) * 1.08),
        size = (500, 300))

    display(plot(pV, pU; layout=(1, 2), size=(900, 300)))
end
