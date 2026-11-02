"""
Numerical experiments for

  B. Sikora, "Rest, final state and delay state: controllability of
  subdiffusion equations with a delay in the diffusion term"
  (submitted to SIAM J. Control Optim.), Section 8.

  d_t^alpha y - y_xx - c y_xx(t-h) = 1_omega u   in (0,1) x (0,T),
  y(0,t) = y(1,t) = 0,  y(0) = y0,  y = phi on [-h,0)
  (y: state, u: control, d_t^alpha: Caputo derivative).

Discretisation: spectral Galerkin (e_n = sqrt(2) sin(n pi x)) and the L1
scheme in time, implicit in -y_xx(t); the delayed term is explicit (h is a
multiple of the time step). No Mittag-Leffler functions are evaluated.

Usage:
  python delay_experiments.py               # everything
  python delay_experiments.py --roots       # Figure 1
  python delay_experiments.py --transport   # Figure 2 (+ errors in Sec. 8.2)
  python delay_experiments.py --control     # Figure 3 and Table 4
  python delay_experiments.py --convergence # Table 3
"""
import numpy as np
import mpmath as mp
from math import gamma
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 11})
ALPHA = 0.75


# ----------------------------------------------------------------------
# 1. Characteristic roots and the neutral chain
# ----------------------------------------------------------------------
def roots_figure(alpha=ALPHA, h=1.0):
    mp.mp.dps = 30
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, c in zip(axes, [0.5, 1.0, 2.0]):
        sig = np.log(abs(c)) / h
        d = 1 if c > 0 else 0
        for n in [1, 2, 3, 5, 10, 30, 100]:
            lam = (n * mp.pi) ** 2
            pts = []
            for k in range(-4, 4):
                sk = (mp.log(abs(c)) + 1j * (2 * k + d) * mp.pi) / h
                if abs(mp.im(sk)) < 1e-12 and mp.re(sk) <= 0:
                    continue                     # not admissible
                guess = sk - sk ** alpha / (h * lam)
                try:
                    r = mp.findroot(lambda s: s ** alpha + lam * (1 + c * mp.e ** (-s * h)), guess)
                except Exception:
                    continue
                res = abs(r ** alpha + lam * (1 + c * mp.e ** (-r * h))) / lam
                if res < 1e-12 and abs(r - sk) < 3.0 and not (mp.im(r) == 0 and mp.re(r) <= 0):
                    pts.append(complex(r))
            pts = np.array(pts)
            if pts.size:
                ax.plot(pts.real, pts.imag, "o", ms=4, label=f"$n={n}$")
        ax.axvline(sig, color="k", ls="--", lw=1)
        ax.axvline(0, color="grey", lw=0.6)
        ax.set_title(rf"$c={c:g}$, $\sigma_c={sig:.2f}$")
        ax.set_xlabel(r"$\mathrm{Re}\,s$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\mathrm{Im}\,s$")
    axes[2].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig("fig_roots.pdf")


# ----------------------------------------------------------------------
# 2. Transport of the history (single mode)
# ----------------------------------------------------------------------
def scalar_l1(lam, alpha, c, h, T, N, u0=0.0, hist=None, f=None):
    """L1 scheme for D^a u + lam u + c lam u(t-h) = f, u(0)=u0, u=hist on [-h,0)."""
    dt = T / N
    t = np.arange(N + 1) * dt
    k = np.arange(N + 1)
    b = (k + 1) ** (1 - alpha) - k ** (1 - alpha)
    C = dt ** (-alpha) / gamma(2 - alpha)
    m = int(round(h / dt))
    u = np.zeros(N + 1); u[0] = u0
    d = np.zeros(N)
    for j in range(1, N + 1):
        histsum = (b[1:j] * d[j - 2::-1][:j - 1]).sum() if j > 1 else 0.0
        ud = u[j - m] if j - m >= 0 else (hist(t[j] - h) if hist else 0.0)
        fj = f(t[j]) if f else 0.0
        u[j] = (C * u[j - 1] - C * histsum - c * lam * ud + fj) / (C + lam)
        d[j - 1] = u[j] - u[j - 1]
    return t, u


def transport_figure(alpha=ALPHA, h=1.0, T=3.0, N=6000):
    g = lambda th: np.sin(np.pi * (th + h) / h)      # history profile on [-h,0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    print("Transport of the history: relative L2 error of lambda*y vs w")
    for ax, c in zip(axes, [0.5, -2.0]):
        for lam in [10.0, 100.0, 1000.0]:
            t, u = scalar_l1(lam, alpha, c, h, T, N, hist=lambda s, lam=lam: g(s) / lam)
            w = lam * u
            ax.plot(t, w, lw=1.2, label=rf"$\lambda={lam:g}$")
        j = np.ceil(t / h).clip(min=1)
        lim = (-c) ** j * g(t - j * h)
        lim[0] = 0.0
        ax.plot(t, lim, "k--", lw=1.2, label=r"limit $w$")
        for lam in [10.0, 100.0, 1000.0]:
            t, u = scalar_l1(lam, alpha, c, h, T, N, hist=lambda s, lam=lam: g(s) / lam)
            err = np.sqrt(np.trapezoid((lam * u - lim) ** 2, t) / np.trapezoid(lim ** 2, t))
            print(f"  c={c:+.1f} lambda={lam:7.0f}: {err:.2e}")
        ax.set_title(rf"$c={c:g}$")
        ax.set_xlabel("$t$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\lambda y(t)$")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("fig_transport.pdf")


# ----------------------------------------------------------------------
# 3. Approximate controls acting in a final window
# ----------------------------------------------------------------------
class DelayProblem:
    def __init__(self, alpha=ALPHA, c=0.0, h=0.5, T=1.0, Nt=400, M=60, Nx=1000,
                 omega=(0.2, 0.4)):
        self.alpha, self.c, self.h, self.T, self.Nt, self.M = alpha, c, h, T, Nt, M
        self.dt = T / Nt
        self.m = int(round(h / self.dt))
        assert abs(self.m * self.dt - h) < 1e-12
        self.x = (np.arange(Nx) + 0.5) / Nx
        self.wx = 1.0 / Nx
        n = np.arange(1, M + 1)
        self.lam = (n * np.pi) ** 2
        self.E = np.sqrt(2.0) * np.sin(np.outer(n, np.pi * self.x))
        self.chi = ((self.x > omega[0]) & (self.x < omega[1])).astype(float)
        self.W = (self.E * self.chi) @ self.E.T * self.wx
        k = np.arange(Nt + 1)
        self.b = (k + 1) ** (1 - alpha) - k ** (1 - alpha)
        self.C = self.dt ** (-alpha) / gamma(2 - alpha)

    def to_modes(self, v):
        return v @ self.E.T * self.wx

    def to_phys(self, a):
        return a @ self.E

    def solve(self, a0, hist_modes, F, c=None, Nsteps=None):
        """Modal L1 scheme with delay. hist_modes(theta) -> modal history.
        F: Nsteps x M modal forcing at t_1..t_N."""
        c = self.c if c is None else c
        N = self.Nt if Nsteps is None else Nsteps
        a = np.zeros((N + 1, self.M)); a[0] = a0
        d = np.zeros((N, self.M))
        for j in range(1, N + 1):
            hs = (self.b[1:j][:, None] * d[j - 2::-1][:j - 1]).sum(axis=0) if j > 1 else 0.0
            ad = a[j - self.m] if j - self.m >= 0 else hist_modes(j * self.dt - self.h)
            a[j] = (self.C * a[j - 1] - self.C * hs - c * self.lam * ad + F[j - 1]) / (self.C + self.lam)
            d[j - 1] = a[j] - a[j - 1]
        return a

    def window_gramian(self, Nw):
        """Discrete control map on a window of Nw steps for c=0 (no delay)."""
        imp = np.zeros((Nw, self.M)); imp[0] = 1.0
        g = self.solve(np.zeros(self.M), lambda th: np.zeros(self.M), imp, c=0.0, Nsteps=Nw)[1:]
        G = g[::-1]                                   # G[j-1] = g_{Nw-j}
        Gram = (G.T @ G) * self.W / self.dt
        Gram = 0.5 * (Gram + Gram.T)
        mu, V = np.linalg.eigh(Gram)
        return G, np.clip(mu, 0, None), V

    def lstar(self, G, phi):
        return self.to_phys(G * phi[None, :]) * self.chi / self.dt


def control_figure(h=0.5, T=1.0, eps_win=0.25):
    y0 = lambda x: np.sin(np.pi * x)
    y1 = lambda x: np.exp(-30 * (x - 0.7) ** 2)
    deltas = 10.0 ** -np.arange(1, 11)
    cs = [0.0, 0.5, 2.0]
    res = {}
    print("Final-window controls (h=%.2f, T=%.2f, window %.2f):" % (h, T, eps_win))
    for c in cs:
        P = DelayProblem(c=c, h=h, T=T)
        a0 = P.to_modes(y0(P.x)); a1 = P.to_modes(y1(P.x))
        hist = lambda th, a0=a0: a0                   # constant history phi = y0
        Nw = int(round(eps_win / P.dt))
        free = P.solve(a0, hist, np.zeros((P.Nt, P.M)))
        p = a1 - free[-1]
        G, mu, V = P.window_gramian(Nw)
        errs, costs, idents, finals = [], [], [], []
        for dl in deltas:
            Rp = V @ ((V.T @ p) / (dl + mu))
            U = np.zeros((P.Nt, P.x.size))
            U[P.Nt - Nw:] = P.lstar(G, Rp)            # control on the final window
            F = P.to_modes(U * P.chi)
            a = P.solve(a0, hist, F)
            err = np.linalg.norm(a[-1] - a1)
            cost = np.sqrt(P.dt * (U ** 2).sum() * P.wx)
            ident = np.linalg.norm(a[-1] - (a1 - dl * Rp))
            errs.append(err); costs.append(cost); idents.append(ident); finals.append(a[-1])
        res[c] = (P, free, errs, costs, finals, a1)
        print(f"  c={c:+.1f}: |y_free(T)|={np.linalg.norm(free[-1]):.3f}  "
              f"max identity defect={max(idents):.1e}")
        for k in [1, 5, 9]:
            print(f"     delta=1e-{k+1}: error={errs[k]:.3e}  cost={costs[k]:.3e}")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    for c in cs:
        P, free, errs, costs, finals, a1 = res[c]
        axes[0].loglog(deltas, errs, "o-", label=rf"$c={c:g}$")
        axes[1].loglog(deltas, costs, "o-", label=rf"$c={c:g}$")
    for ax in axes[:2]:
        ax.invert_xaxis(); ax.grid(True, which="both", alpha=0.3); ax.legend()
        ax.set_xlabel(r"$\delta$")
    axes[0].set_ylabel(r"$\|y_\delta(T)-y_1\|$")
    axes[1].set_ylabel(r"$\|u_\delta\|_{L^2(\omega\times(T-\varepsilon,T))}$")
    P, free, errs, costs, finals, a1 = res[2.0]
    ax = axes[2]
    ax.plot(P.x, P.to_phys(a1), "k--", lw=2, label=r"target $y_1$")
    for k in [1, 5, 9]:
        ax.plot(P.x, P.to_phys(finals[k]), label=rf"$\delta=10^{{-{k+1}}}$")
    yf = P.to_phys(free[-1]); s = np.abs(yf).max()
    ax.plot(P.x, yf / s, ":", label=rf"free $y(T)$, scaled by $1/{s:.1f}$")
    ax.axvspan(0.2, 0.4, color="grey", alpha=0.2, label=r"$\omega$")
    ax.set_title(r"$c=2$"); ax.set_xlabel("$x$"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_delay_control.pdf")


# ----------------------------------------------------------------------
# 4. Grid convergence for the final-window controls (c = 2)
# ----------------------------------------------------------------------
def final_window_run(c, Nt, M, h=0.5, T=1.0, eps_win=0.25, deltas=(1e-2, 1e-6, 1e-10)):
    y0 = lambda x: np.sin(np.pi * x)
    y1 = lambda x: np.exp(-30 * (x - 0.7) ** 2)
    P = DelayProblem(c=c, h=h, T=T, Nt=Nt, M=M)
    a0 = P.to_modes(y0(P.x)); a1 = P.to_modes(y1(P.x))
    hist = lambda th, a0=a0: a0
    Nw = int(round(eps_win / P.dt))
    free = P.solve(a0, hist, np.zeros((P.Nt, P.M)))
    p = a1 - free[-1]
    G, mu, V = P.window_gramian(Nw)
    out = []
    for dl in deltas:
        Rp = V @ ((V.T @ p) / (dl + mu))
        U = np.zeros((P.Nt, P.x.size)); U[P.Nt - Nw:] = P.lstar(G, Rp)
        a = P.solve(a0, hist, P.to_modes(U * P.chi))
        out.append((np.linalg.norm(a[-1] - a1), np.sqrt(P.dt * (U ** 2).sum() * P.wx)))
    return np.linalg.norm(free[-1]), out


def convergence_table(c=2.0):
    print("Grid convergence, c=%g:" % c)
    rows = []
    for Nt, M in [(400, 60), (800, 60), (400, 120), (800, 120), (1600, 120)]:
        nf, out = final_window_run(c, Nt, M)
        line = f"{Nt} & {M} & {nf:.3f} & " + " & ".join(f"{e:.2e} & {k:.2e}" for e, k in out) + r" \\"
        print("  " + line, flush=True)
        rows.append(line)
    with open("table_delay_convergence.tex", "w") as fh:
        fh.write("\n".join(rows) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ["roots", "transport", "control", "convergence"]:
        ap.add_argument(f"--{name}", action="store_true")
    args = ap.parse_args()
    run_all = not any(vars(args).values())
    if run_all or args.roots:
        roots_figure()
    if run_all or args.transport:
        transport_figure()
    if run_all or args.control:
        control_figure()
    if run_all or args.convergence:
        convergence_table()


if __name__ == "__main__":
    main()
