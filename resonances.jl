###############################################
# resonances.jl — Resonances of H = -∂ₓ² + V(x)
#                 on the half-line (0, ∞)
###############################################
#
# Finds resonances of H = -∂ₓ² + V(x) with
#   BC at x = 0: u(0) = 0            (Dirichlet)
#   BC at x = b: u'(b) = ik u(b)    (outgoing / resonance)
#
# Since V is compactly supported on [0, b], the outgoing condition
# at x = b is exact (the free solution beyond b is e^{ikx}).
#
# Parameterization: k = iκ, κ ∈ ℝ.
#   κ > 0 → bound state    (Im(k) > 0, u decays as e^{-κx})
#   κ < 0 → resonance      (Im(k) < 0, u grows as e^{|κ|x})
#
# Method: central-difference ghost-point elimination of u_{N+1}
# converts the resonance BC at x = b into a quadratic EVP (QEP):
#
#   (k² I + ik B - A) u = 0
#
# where A is the modified tridiagonal (2nd-order FD for -∂ₓ² + V,
# with the last subdiagonal entry doubled) and B = (2/h) eₙ eₙᵀ
# (rank-1, nonzero only at the N,N entry).
#
# The QEP is linearized to the 2N×2N companion EVP
#   C [u; k u] = k [u; k u],   C = [ 0,  I ]
#                                    [ A, -iB ]
# whose eigenvalues are the resonances k.
#
# For each resonance with Im(k) < 0 and |k| < k_max, we set
#   κ = -ik
# and compute
#   c = -4κ ∫₀ᵇ (U(x)/U(b))⁴ dx
# where U is the resonance eigenfunction (normalized so U(b) = 1).
###############################################

using LinearAlgebra

"""
    compute_resonances(b, Vfun; N=300, k_max=10.0, im_tol=-1e-4)

Find resonances of H = -∂ₓ² + V(x) on the half-line (0, ∞).

The problem is posed on [0, b] with Dirichlet BC at x = 0 and outgoing
resonance BC u'(b) = iku(b) at x = b. A central-difference ghost-point
elimination converts this to a quadratic EVP, which is linearized to a
2N×2N companion eigenvalue problem and solved with `eigen`.

# Grid
N interior/boundary nodes at xⱼ = j·h, j = 1…N, h = b/N (so xₙ = b).
The Dirichlet condition u₀ = 0 is built in; uₙ = u(b) is an unknown.

# Derivation of the companion
Ghost-point elimination: u'(b) ≈ (u_{N+1} − u_{N−1})/(2h) = iku_N
  → u_{N+1} = u_{N−1} + 2ihk uₙ

Substituting into the centered FD at j = N gives row N of A as
  A[N, N−1] = −2/h²,   A[N, N] = 2/h² + V(b)
plus a k-dependent term −(2ik/h) uₙ, captured by B[N,N] = 2/h.

QEP: (k²I + ikB − A)u = 0
Companion C = [0, I; A, −iB] satisfies C[u; ku] = k[u; ku].

# Arguments
- `b`:      support boundary (V(x) = 0 for x > b)
- `Vfun`:   potential function V : ℝ → ℝ
- `N`:      number of FD nodes (matrix size 2N × 2N; default 300)
- `k_max`:  discard resonances with |k| ≥ k_max (default 10.0)
- `im_tol`: discard eigenvalues with Im(k) ≥ im_tol (default −1e-4)

# Returns
Vector of NamedTuples, sorted by Im(k) (most negative first):
  `k`  — complex resonance wavenumber
  `κ`  — −ik  (real when k is purely imaginary)
  `U`  — resonance eigenfunction on x₁…xₙ, normalized so U(b) = 1
  `x`  — FD grid [h, 2h, …, b]
  `I`  — ∫₀ᵇ (U(x)/U(b))⁴ dx  (trapezoidal, using u(0)=0)
  `c`  — −4κ · I
"""
function compute_resonances(b, Vfun; N=300, k_max=10.0, im_tol=-1e-4)
    h     = b / N
    xgrid = [j * h for j in 1:N]   # x₁, …, xₙ  (xₙ = b)

    # ── Build modified stiffness matrix A ────────────────────────
    # A is N×N.  Rows j = 1 … N−1: standard centered FD.
    # Row j = N: ghost-point elimination doubles the subdiagonal.
    A = zeros(ComplexF64, N, N)
    for j in 1:N
        A[j, j] = 2.0/h^2 + Vfun(xgrid[j])
        if j > 1
            A[j, j-1] = (j == N) ? -2.0/h^2 : -1.0/h^2
        end
        if j < N
            A[j, j+1] = -1.0/h^2
        end
    end

    # ── Linear-in-k matrix B  (rank-1, only (N,N) nonzero) ──────
    B = zeros(ComplexF64, N, N)
    B[N, N] = 2.0 / h

    # ── Companion  C = [0, I; A, −iB] ───────────────────────────
    # QEP (k²I + ikB − A)u = 0  →  k²u = Au − ikBu
    # With z = ku: kz = Az − iBz, so C = [0, I; A, −iB].
    C = zeros(ComplexF64, 2N, 2N)
    for j in 1:N
        C[j, N+j] = 1.0                 # upper-right: identity
    end
    C[N+1:2N, 1:N]     .=  A           # lower-left:   A
    C[N+1:2N, N+1:2N]  .= -im .* B    # lower-right: −iB

    # ── Solve the full 2N×2N EVP ─────────────────────────────────
    F       = eigen(C)
    all_k   = F.values
    all_vec = F.vectors

    # ── Filter and assemble output ────────────────────────────────
    result = []
    for (j, k) in enumerate(all_k)
        imag(k) >= im_tol && continue    # exclude bound states / real axis
        abs(k)  >  k_max  && continue    # outside requested window

        # Resonance eigenfunction: upper N components of eigenvector
        u_raw = all_vec[1:N, j]

        # Skip if u(b) is numerically zero (degenerate mode)
        u_b = u_raw[N]
        abs(u_b) < 1e-10 * norm(u_raw) && continue

        # Normalize so U(b) = 1
        U = u_raw ./ u_b

        # Trapezoidal integral ∫₀ᵇ (U(x)/U(b))⁴ dx
        # Nodes: x₀=0 with u=0  (contributes 0), then x₁…xₙ.
        # Rule:  h · [½f₀ + f₁ + … + f_{N−1} + ½fₙ]
        #      = h · [f₁ + … + f_{N−1} + ½]   since f₀=0, fₙ=1
        f     = U .^ 4
        I_val = h * (sum(f[1:N-1]) + 0.5 * f[N])

        κ = -im * k      # κ = −ik  (real when k purely imaginary)
        c = -4 * κ * I_val

        push!(result, (; k=k, κ=κ, U=U, x=xgrid, I=I_val, c=c))
    end

    sort!(result; by = r -> imag(r.k))
    return result
end


"""
    print_resonances(resonances)

Pretty-print the resonance table returned by `compute_resonances`.
"""
function print_resonances(resonances)
    if isempty(resonances)
        println("  No resonances found in the specified window.")
        return
    end

    println("  Found $(length(resonances)) resonance(s):\n")
    println("  " * "─"^88)
    @printf("  %4s  %24s  %20s  %20s  %20s\n",
            "#", "k", "κ = −ik", "I = ∫(U/U(b))⁴dx", "c = −4κ·I")
    println("  " * "─"^88)

    for (idx, r) in enumerate(resonances)
        # Format complex numbers as (re + im·i)
        @printf("  %4d  %+10.5f %+10.5f i  %+8.4f %+8.4f i  %+8.4f %+8.4f i  %+8.4f %+8.4f i\n",
                idx,
                real(r.k),  imag(r.k),
                real(r.κ),  imag(r.κ),
                real(r.I),  imag(r.I),
                real(r.c),  imag(r.c))
    end
    println("  " * "─"^88)
    println()
    println("  Note: for resonances on the imaginary k-axis (Re(k) ≈ 0),")
    println("        κ = −ik is real and negative, and c is real.")
end
