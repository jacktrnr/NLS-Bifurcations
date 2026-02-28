potential_type = :smooth_bump # :smooth, :threestep, :gaussian, :step, :square, :square_bump, :smooth_bump
b = 2.0                  # Support boundary: V(x) = 0 for x > b
V0 = -2.0 # Potential depth/height (negative = attractive well)

# Type-specific parameters (ignored if not relevant to chosen type):
bump_amp_factor = 2.3    # :square_bump — bump amplitude = factor * |V0|
bump_width_frac = 0.1    # :square_bump — bump width = fraction * b
bump_center_frac = 1.0   # :square_bump — bump center = fraction * b
V1 = -6.0               # :step — left-half depth
σ_frac = 0.25           # :gaussian — width σ = fraction * b
edge_height = 0.1       # :threestep — height of edge regions
b1_frac = 0.9           # :smooth_bump — well/bump transition point = b1_frac * b
bump_height = 0.1       # :smooth_bump — peak height of the smooth bump piece (positive)

println("="^70)
println("HALF-LINE NLS BIFURCATION ANALYSIS")
println("="^70)
println("  Equation : -ψ'' + V(x)ψ - ψ³ = Eψ  on x > 0")
println("  BC       : ψ(0) = 0")
println("  Potential: $potential_type  (b = $b, V0 = $V0)")
println("="^70)

# Build potential via dispatcher
Vfun = make_potential(potential_type;
    b=b, V0=V0,
    bump_amp_factor=bump_amp_factor,
    bump_width_frac=bump_width_frac,
    bump_center_frac=bump_center_frac,
    V1=V1, σ_frac=σ_frac, edge_height=edge_height,
    b1_frac=b1_frac, bump_height=bump_height)

label = potential_label(potential_type; b=b, V0=V0)
println("  Label    : $label")
println()

# Build V(x) panel (same style as plot_potential, without auto-display)
_xmax_V = max(3*b, 5.0)
_xs_V   = range(0, _xmax_V; length=500)
_Vs     = [Vfun(x) for x in _xs_V]
_plt_V  = plot(_xs_V, _Vs; color=:red, lw=2.5, xlabel=L"x", ylabel=L"V(x)", label="",
               size=(500, 300))
vline!(_plt_V, [b]; color=:gray40, ls=:dash, lw=1.5, alpha=0.6, label=L"x = b")
hline!(_plt_V, [0.0]; color=:black, ls=:dot, lw=1, alpha=0.3, label="")


println("\n" * "="^70)
println("RESONANCES OF -∂ₓ^2 + V(x)")
println("="^70)
println("  Method : quadratic EVP → 2N×2N companion (N = $res_N)")
println("  Filter : Im(k) < 0,  |k| < $res_k_max")
println("  Potential: $potential_type  (b = $b, V0 = $V0)")
println()

t_res = @elapsed begin
    resonances = compute_resonances(b, Vfun; N=res_N, k_max=res_k_max)
end

print_resonances(resonances)
@printf("  (computed in %.2f seconds)\n", t_res)

# ── U_★ for each negative-imaginary resonance ────────────────────────────────
_res_imag = filter(r -> imag(r.k) < 0 && abs(real(r.k)) < 1e-6, resonances)

if isempty(_res_imag)
    println("  No negative-imaginary resonances found — no U_★ to plot.")
else
    println("\n" * "="^70)
    println("U_★  (resonance modes, ordered by γ)")
    println("="^70)

    for r in sort(_res_imag; by = r -> abs(imag(r.k)))
        γ_r = -imag(r.k)
        E_r = -γ_r^2

        @printf("  γ = %.6f   E_bif = %.6f\n", γ_r, E_r)

        sol_u = solve(
            ODEProblem(
                (du, u, p, x) -> (du[1] = u[2]; du[2] = (Vfun(x) + γ_r^2) * u[1]),
                [0.0, 1.0], (0.0, b)),
            Tsit5(); reltol=1e-13, abstol=1e-15,
            saveat=range(0.0, b; length=2001))

        xu = sol_u.t
        Uu = [u[1] for u in sol_u.u]

        p_u = plot(xu, Uu;
            xlabel = L"x",
            ylabel = L"U_{\star}(x)",
            label  = false,
            lw     = 2.5,
            xlims  = (0, b),
            ylims  = (0, max(maximum(Uu), 0) * 1.08),
            title = "γ = $(round(γ_r, sigdigits=4))",
            size   = (500, 300))

        display(plot(_plt_V, p_u; layout=(1, 2), size=(900, 300)))
    end
end

