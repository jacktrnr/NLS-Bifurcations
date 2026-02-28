###############################################################
# findV.jl
#
# Direct scan: for each a in [1, 2], find all resonances γ
# satisfying F(γ, a) = U'(1) - γ U(1) = 0
# where U solves  -U'' + (ax + c + γ²)U = 0, U(0)=0, U'(0)=1
# with c = -1 fixed on [0, 1].
#
# For each (a, γ) pair, compute dN/dE via the formula:
#   Ω     = U(1)⁴/(2γ) - 2 ∫₀¹ U⁴ dx
#   A     = ∫₀¹ U² dx - U(1)²/(2γ)
#   dN/dE = -2/γ + 2A²/Ω
#
# Resonances are grouped into branches by sorting γ at each a.
# Sign changes of dN/dE along each branch are located and refined.
###############################################################

using OrdinaryDiffEq
using Roots
using Plots
using LaTeXStrings
using Printf

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
const c_fixed = -1.0
const b_end   = 1.0

const a_start = 1.0
const a_end   = 1.1
const n_a     = 300        # number of a values in scan

const γ_lo    = 0.1        # lower bound for γ scan
const γ_hi    = 15.0       # upper bound for γ scan
const n_γ     = 1000       # scan resolution (points per a value)
const n_pts   = 2001       # ODE grid points for diagnostics integrals

# -----------------------------------------------------------------------
# ODE shooter
# -----------------------------------------------------------------------
function shoot(γ::Real, a::Real)
    rhs!(du, u, p, x) = (du[1] = u[2]; du[2] = (a*x + c_fixed + γ^2)*u[1])
    sol = solve(ODEProblem(rhs!, [0.0, 1.0], (0.0, b_end)),
                Tsit5(); abstol=1e-14, reltol=1e-14, save_everystep=false)
    return sol.u[end]   # [U(1), U'(1)]
end

F_res(γ, a) = (u = shoot(γ, a); u[2] - γ*u[1])

# Dense solve for integrals
function solve_Ustar_dense(γ, a)
    rhs!(du, u, p, x) = (du[1] = u[2]; du[2] = (a*x + c_fixed + γ^2)*u[1])
    return solve(ODEProblem(rhs!, [0.0, 1.0], (0.0, b_end)),
                 Tsit5(); abstol=1e-14, reltol=1e-14,
                 saveat=range(0.0, b_end; length=n_pts))
end

# -----------------------------------------------------------------------
# Find all resonances for a given a via grid scan + bisection
# -----------------------------------------------------------------------
function find_all_resonances(a)
    γs = range(γ_lo, γ_hi; length=n_γ)
    Fs = [F_res(γ, a) for γ in γs]
    roots = Float64[]
    for i in 1:n_γ-1
        isfinite(Fs[i]) && isfinite(Fs[i+1]) && Fs[i]*Fs[i+1] < 0 || continue
        γ_root = find_zero(γ -> F_res(γ, a), (γs[i], γs[i+1]), Bisection(); atol=1e-13)
        push!(roots, γ_root)
    end
    return roots   # sorted increasing by construction
end

# -----------------------------------------------------------------------
# Diagnostics: Ω, A, dN/dE at a given (γ, a) point
# -----------------------------------------------------------------------
function diagnostics(γ, a)
    sol  = solve_Ustar_dense(γ, a)
    ts   = sol.t
    Us   = [s[1] for s in sol.u]
    dx   = ts[2] - ts[1]
    int4 = dx * (sum(Us.^4) - 0.5*(Us[1]^4 + Us[end]^4))
    int2 = dx * (sum(Us.^2) - 0.5*(Us[1]^2 + Us[end]^2))
    U1    = Us[end]
    Omega = U1^4 / (2γ) - 2*int4
    A     = int2 - U1^2 / (2γ)
    dNdE  = abs(Omega) > 1e-15 ? -2/γ + 2*A^2/Omega : NaN
    # dN/dE = 0  ⟺  Ω = γA²  (smooth, no pole)
    res   = Omega - γ * A^2
    pos   = all(u > -1e-10 for u in Us[2:end-1])
    return (Omega=Omega, A=A, dNdE=dNdE, res=res, U1=U1, int2=int2, int4=int4,
            Us=Us, ts=ts, pos=pos)
end

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
let

println("="^70)
println("  findV.jl  —  direct scan of resonance branches")
println("  V(x) = a·x + ($c_fixed)  on [0, $b_end]")
@printf("  a: %.1f → %.1f,  n_a = %d,  γ ∈ [%.1f, %.1f]\n",
        a_start, a_end, n_a, γ_lo, γ_hi)
println("="^70)

a_vals = collect(range(a_start, a_end; length=n_a))

# ── 1. Scan: collect all (a, γ, dN/dE, Ω) data ────────────────────────
println("\nScanning resonances ...")

# branch_data[k] = vectors for the k-th smallest resonance at each a
max_branches = 5
branch_a    = [Float64[] for _ in 1:max_branches]
branch_γ    = [Float64[] for _ in 1:max_branches]
branch_dNdE = [Float64[] for _ in 1:max_branches]
branch_Ω    = [Float64[] for _ in 1:max_branches]
branch_res  = [Float64[] for _ in 1:max_branches]   # Ω - γA² (smooth proxy for dN/dE=0)

for (ia, a) in enumerate(a_vals)
    γs = find_all_resonances(a)
    for (k, γ) in enumerate(γs)
        k > max_branches && break
        d = diagnostics(γ, a)
        push!(branch_a[k],    a)
        push!(branch_γ[k],    γ)
        push!(branch_dNdE[k], d.dNdE)
        push!(branch_Ω[k],    d.Omega)
        push!(branch_res[k],  d.res)
    end
    ia % 50 == 0 &&
        @printf("  a = %.3f:  %d resonance(s) found  (γ ∈ [%.3f, %.3f])\n",
                a, length(γs),
                isempty(γs) ? NaN : γs[1],
                isempty(γs) ? NaN : γs[end])
end

println("\nScan complete.")
for k in 1:max_branches
    isempty(branch_a[k]) && continue
    @printf("  Branch %d: %d points,  a ∈ [%.4f, %.4f],  γ ∈ [%.4f, %.4f]\n",
            k, length(branch_a[k]),
            minimum(branch_a[k]), maximum(branch_a[k]),
            minimum(branch_γ[k]), maximum(branch_γ[k]))
end

# ── 2. Find dN/dE = 0 crossings using the smooth residual Ω − γA² ─────
# dN/dE = 0  ⟺  Ω = γA²  ⟺  res = Ω − γA² = 0
# This avoids the pole in dN/dE at Ω=0.
println("\n===== Zeros of dN/dE  (via residual Ω − γA² = 0) =====")
zero_crossings = []

for k in 1:max_branches
    av  = branch_a[k]
    γv  = branch_γ[k]
    rv  = branch_res[k]   # smooth: Ω − γA²
    isempty(av) && continue

    for i in 1:length(av)-1
        isnan(rv[i]) || isnan(rv[i+1]) && continue
        rv[i] * rv[i+1] >= 0 && continue   # no sign change

        # Bisect in a on the resonance branch: find where Ω − γA² = 0
        function res_at_a(a_s)
            γ_mid = (γv[i] + γv[i+1]) / 2
            hw    = max(0.5, abs(γv[i+1] - γv[i]) + 0.3)
            γ_lo2 = max(γ_lo, γ_mid - hw)
            γ_hi2 = min(γ_hi, γ_mid + hw)
            fa    = F_res(γ_lo2, a_s)
            fb    = F_res(γ_hi2, a_s)
            γ_s   = (fa*fb < 0) ?
                find_zero(γ -> F_res(γ, a_s), (γ_lo2, γ_hi2), Bisection(); atol=1e-13) :
                γ_mid
            return diagnostics(γ_s, a_s).res
        end

        a_star = try
            find_zero(res_at_a, (av[i], av[i+1]), Bisection(); atol=1e-12)
        catch
            av[i] - rv[i]*(av[i+1]-av[i])/(rv[i+1]-rv[i])   # linear fallback
        end

        # Final γ at a_star
        γ_mid = (γv[i]+γv[i+1])/2
        hw    = max(0.5, abs(γv[i+1]-γv[i]) + 0.3)
        γ_star = try
            find_zero(γ -> F_res(γ, a_star),
                      (max(γ_lo, γ_mid-hw), min(γ_hi, γ_mid+hw)),
                      Bisection(); atol=1e-13)
        catch; γ_mid end

        d = diagnostics(γ_star, a_star)
        push!(zero_crossings, (branch=k, a_star=a_star, γ_star=γ_star, d=d))

        @printf("  Branch %d crossing:\n", k)
        @printf("    a★     = %.12f\n",     a_star)
        @printf("    γ★     = %.12f\n",     γ_star)
        @printf("    dN/dE  = %+.6e  (should be ≈ 0)\n", d.dNdE)
        @printf("    Ω      = %.10f\n",     d.Omega)
        @printf("    γA²    = %.10f\n",     γ_star * d.A^2)
        @printf("    A      = %.10f\n",     d.A)
        @printf("    U(1)   = %.10f\n",     d.U1)
        @printf("    U★>0:    %s\n",        d.pos)
        ν_0 = 3*d.Omega / (4γ_star)
        @printf("    ν₀     = %.8f\n\n",   ν_0)
    end
end

isempty(zero_crossings) && println("  No dN/dE = 0 crossings found on any branch.")

# ── 3. Plots ───────────────────────────────────────────────────────────
colors = [:blue, :red, :green4, :purple, :orange]
labels = ["Branch $k" for k in 1:max_branches]

pγ   = plot(; xlabel=L"a", ylabel=L"\gamma",           legend=:topleft)
pres = plot(; xlabel=L"a", ylabel=L"\Omega - \gamma A^2", legend=:best)
pΩ   = plot(; xlabel=L"a", ylabel=L"\Omega",           legend=:best)
hline!(pres, [0.0]; ls=:dash, color=:black, lw=1.5, label=L"\Omega=\gamma A^2")
hline!(pΩ,   [0.0]; ls=:dash, color=:black, lw=1.5, label=L"\Omega=0")

for k in 1:max_branches
    isempty(branch_a[k]) && continue
    col = colors[k]
    lbl = labels[k]

    plot!(pγ, branch_a[k], branch_γ[k];
          lw=2, color=col, label=lbl)

    # res = Ω − γA² is smooth, no filtering needed
    plot!(pres, branch_a[k], branch_res[k];
          lw=2, color=col, label=lbl)

    mask_Ω = isfinite.(branch_Ω[k]) .& (abs.(branch_Ω[k]) .< 1.0)
    if any(mask_Ω)
        plot!(pΩ, branch_a[k][mask_Ω], branch_Ω[k][mask_Ω];
              lw=2, color=col, label=lbl)
    end
end

for zc in zero_crossings
    scatter!(pγ,   [zc.a_star], [zc.γ_star]; ms=8, color=:black, label="")
    scatter!(pres, [zc.a_star], [0.0];        ms=8, color=:black, label="")
    scatter!(pΩ,   [zc.a_star], [zc.d.Omega]; ms=8, color=:black, label="")
end

fig = plot(pγ, pres, pΩ; layout=(3,1), size=(800,1050), left_margin=5Plots.mm)
display(fig)
savefig(fig, "findV_branches.png")
println("Figure saved to findV_branches.png")

# ── 4. V(x) and U★(x) at each dN/dE = 0 point ───────────────────────
for zc in zero_crossings
    xs_V = range(0.0, b_end; length=500)
    Vs   = @. zc.a_star * xs_V + c_fixed

    pV = plot(xs_V, Vs;
        xlabel=L"x", ylabel=L"V(x)",
        title=@sprintf("a★=%.8f, γ★=%.8f", zc.a_star, zc.γ_star),
        label=false, lw=2.5, color=:red)
    hline!(pV, [0.0]; ls=:dot, color=:black, lw=1, label="")

    pU = plot(zc.d.ts, zc.d.Us;
        xlabel=L"x", ylabel=L"U_\star(x)",
        label=false, lw=2.5, color=:blue,
        xlims=(0, b_end),
        ylims=(0, max(maximum(zc.d.Us), 0)*1.08))

    display(plot(pV, pU; layout=(1,2), size=(900,350)))
end

end # let
