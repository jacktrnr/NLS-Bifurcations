###############################################################
# check_jost.jl
# Verifies the Pöschl-Teller Jost solution f₊ on x > b.
#
# For x > b the background is the soliton S(x-y; E) and the
# linearized eigenvalue equations reduce to the PT problem
#   -φ'' - ℓ(ℓ+1)κ²sech²(κ(x-y))φ = (μ-κ²)φ       (*)
# with κ = √(-E), where ℓ=2 for L₊ and ℓ=1 for L₋.
#
# The exact L² solution is the Jost function (Prop. 2.1 of paper):
#   ℓ=2: f₊(ξ;λ) = e^{-λξ}(m²-1+3mt+3t²)/((m+1)(m+2))
#   ℓ=1: f₊(ξ;λ) = e^{-λξ}(m+t)/(m+1)
# with ξ=x-y, t=tanh(κξ), m=λ/κ, λ=√(κ²-μ).
#
# Three checks:
#   1. ODE residual: f₊ pointwise satisfies (*) (finite-difference)
#   2. ODE integration: integrating (*) from x=y (ξ=0) with Jost IC
#      reproduces the Jost formula to ODE tolerance
#   3. Log-derivative: the formula w_out = f₊'/f₊|_{x=b} from
#      eq. (5.14) matches the analytic derivative of f₊
###############################################################

using OrdinaryDiffEq

# ── Choose parameters freely ────────────────────────────────
E  = -2.0          # energy (< 0)
y  = 8.0           # soliton centre (> b; otherwise arbitrary)
b  = 1.0           # support edge
μp = -7.0          # spectral parameter for L₊  (need μ < κ² = -E)
μm = -0.3          # spectral parameter for L₋

κ  = sqrt(-E)

println("=" ^ 60)
println("PT JOST CHECK   E=$E  κ=$(round(κ,digits=4))  y=$y  b=$b")
println("=" ^ 60)

# ── Jost functions (analytic) ───────────────────────────────
function jost(ξ, κ, λ, ell)
    m = λ / κ
    t = tanh(κ * ξ)
    if ell == 2
        P = m^2 - 1 + 3m*t + 3t^2
        return exp(-λ*ξ) * P / ((m+1)*(m+2))
    else   # ell == 1
        return exp(-λ*ξ) * (m + t) / (m+1)
    end
end

function jost_deriv(ξ, κ, λ, ell)
    m  = λ / κ
    t  = tanh(κ * ξ)
    s2 = 1 - t^2           # sech²
    if ell == 2
        P  = m^2 - 1 + 3m*t + 3t^2
        dP = (3m + 6t) * κ * s2
        return exp(-λ*ξ) * (-λ*P + dP) / ((m+1)*(m+2))
    else
        return exp(-λ*ξ) * (-λ*(m+t) + κ*s2) / (m+1)
    end
end

# ── Run both ℓ checks ───────────────────────────────────────
for (ell, μ, lbl) in [(2, μp, "L₊  ℓ=2"), (1, μm, "L₋  ℓ=1")]

    λ = sqrt(κ^2 - μ)
    m = λ / κ
    u = tanh(κ*(b - y))    # tanh(κ(b-y)), negative since b < y
    println("\n── $lbl   μ=$μ   λ=$(round(λ,digits=6))   u=$(round(u,digits=6)) ──")

    # ── Check 1: pointwise ODE residual via 2nd-order FD ───
    # Start from x=y (ξ=0) to stay in the decaying regime of f₊.
    # For x < y the Jost function grows like e^{λ(y-x)}, so absolute
    # residuals become huge even when the relative error is tiny.
    xs  = range(y, y + 20/λ; length=2000)
    h   = step(xs)
    fv  = [jost(x - y, κ, λ, ell) for x in xs]
    f″  = [(fv[i+1] - 2fv[i] + fv[i-1]) / h^2 for i in 2:length(xs)-1]
    xm  = xs[2:end-1]
    pot = ell*(ell+1) * κ^2 .* sech.(κ .* (xm .- y)).^2
    res = -f″ .- pot .* fv[2:end-1] .- (μ - κ^2) .* fv[2:end-1]
    println("  1. FD residual  max|R|  = $(round(maximum(abs.(res)), sigdigits=3))")

    # ── Check 2: integrate ODE from x=y, compare to formula ─
    # Start from y (ξ=0) for the same reason as Check 1.
    f0  = jost(0.0, κ, λ, ell)
    df0 = jost_deriv(0.0, κ, λ, ell)
    function rhs!(du, u, p, x)
        V = ell*(ell+1) * κ^2 * sech(κ*(x - y))^2
        du[1] = u[2]
        du[2] = (-V - (μ - κ^2)) * u[1]   # = (V_pt - λ²) φ
    end
    sol = solve(ODEProblem(rhs!, [f0, df0], (y, y + 20/λ)), Tsit5();
                reltol=1e-13, abstol=1e-15,
                saveat=range(y, y + 20/λ; length=500))
    φ_num   = [u[1] for u in sol.u]
    φ_exact = [jost(x - y, κ, λ, ell) for x in sol.t]
    err_ode = maximum(abs.(φ_num .- φ_exact))
    println("  2. ODE match    max|num-exact| = $(round(err_ode, sigdigits=3))")

    # ── Check 3: log-derivative formula matches f₊'/f₊ at x=b
    w_exact = jost_deriv(b - y, κ, λ, ell) / jost(b - y, κ, λ, ell)
    if ell == 2
        N = m^2 - 1 + 3m*u + 3u^2
        w_formula = κ * (-m*N + (1 - u^2)*(3m + 6u)) / N
    else
        w_formula = κ * (-m + (1 - u^2)/(m + u))
    end
    println("  3. Log-deriv    formula=$(round(w_formula,digits=8))  " *
            "exact=$(round(w_exact,digits=8))  " *
            "err=$(round(abs(w_formula-w_exact), sigdigits=3))")
end

println("\n" * "=" ^ 60)
println("Checks 1,2: expect O(h²) and O(ODE tol).  Check 3: should be exact.")
println("=" ^ 60)
