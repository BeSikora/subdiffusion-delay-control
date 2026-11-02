"""
Numerical checks of identities quoted in the text of

  B. Sikora, "Rest, final state and delay state: controllability of
  subdiffusion equations with a delay in the diffusion term".

  python checks.py

  (1) Remark 3.6(a): accuracy of the first-order formula for the zeros of
      Delta_n near the neutral chain (n = 100);
  (2) Lemma 5.3: duality identity, single mode, L1 scheme;
  (3) Lemma 5.7: Bernstein representation of k_lambda;
  (4) Proposition 5.9: product identity for rational alpha = p/q;
  (5) Lemma 6.1: controls acting in the final window do not see the delay.
"""
import cmath
import math
from math import gamma

import mpmath as mp
import numpy as np
from scipy.integrate import quad, trapezoid


def check_chain(alpha=0.75, h=1.0, n=100, k=1):
    mp.mp.dps = 30
    lam = (n * mp.pi) ** 2
    worst = 0.0
    for c in [2.0, -2.0, 1.0, 0.5, -0.5]:
        d = 1 if c > 0 else 0
        sk = (mp.log(abs(c)) + 1j * (2 * k + d) * mp.pi) / h
        guess = sk - sk ** alpha / (h * lam)
        r = mp.findroot(lambda s: s ** alpha + lam * (1 + c * mp.e ** (-s * h)), guess)
        worst = max(worst, float(abs(r - guess)))
    print(f"(1) first-order formula, n={n}: max error {worst:.1e}")


def check_duality(alpha=0.75, lam=7.0, T=1.0, N=8000):
    dt = T / N
    t = np.arange(N + 1) * dt
    k = np.arange(N + 1)
    b = (k + 1) ** (1 - alpha) - k ** (1 - alpha)
    C = dt ** (-alpha) / gamma(2 - alpha)
    f = lambda s: np.cos(3 * s) + s
    y = np.zeros(N + 1)
    d = np.zeros(N)
    for j in range(1, N + 1):
        hist = (b[1:j] * d[j - 2::-1][:j - 1]).sum() if j > 1 else 0.0
        y[j] = (C * y[j - 1] - C * hist + f(t[j])) / (C + lam)
        d[j - 1] = y[j] - y[j - 1]

    def a(s):
        x = (s - 0.5) / 0.4
        return math.exp(-1 / (1 - x * x)) if abs(x) < 1 else 0.0

    def ap(s, e=1e-7):
        return (a(s + e) - a(s - e)) / (2 * e)

    def Dback(ti):  # backward Caputo derivative of a
        if ti >= 0.9:
            return 0.0
        if ti < 0.1:
            val, _ = quad(lambda s: (s - ti) ** (-alpha) * ap(s), 0.1, 0.9, limit=200)
        else:
            val, _ = quad(ap, ti, 0.9, weight="alg", wvar=(-alpha, 0), limit=200)
        return -val / gamma(1 - alpha)

    tt = t[::20]
    D = np.array([Dback(x) for x in tt])
    A = np.array([a(x) for x in tt])
    lhs = trapezoid(np.interp(tt, t, y) * (D + lam * A), tt)
    rhs = trapezoid(f(t) * np.array([a(x) for x in t]), t)
    print(f"(2) duality identity: relative difference {abs(lhs - rhs) / abs(rhs):.1e}")


def check_bernstein(lam=4.0):
    mp.mp.dps = 25
    worst = 0.0
    for a in [mp.mpf("0.75"), mp.mpf(2) / 3, mp.mpf("0.7725")]:
        K = lambda r: (mp.sin(a * mp.pi) / mp.pi * r ** a
                       / (r ** (2 * a) + 2 * lam * r ** a * mp.cos(a * mp.pi) + lam ** 2))
        for t in [mp.mpf("0.3"), mp.mpf("1.2")]:
            lhs = t ** (a - 1) * mp.nsum(lambda k: (-lam * t ** a) ** k / mp.gamma(a * k + a), [0, mp.inf])
            rhs = mp.quad(lambda r: mp.e ** (-r * t) * K(r), [0, 1, 10, mp.inf])
            worst = max(worst, float(abs(lhs - rhs) / abs(lhs)))
    print(f"(3) Bernstein representation: max relative error {worst:.1e}")


def check_product(trials=20, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for (p, q) in [(2, 3), (3, 4), (5, 7)]:
        a = p / q
        sigma = (-1) ** (p + q)
        for _ in range(trials):
            r, lam = rng.uniform(0.2, 5), rng.uniform(1, 30)
            for sgn in (1, -1):
                ph = cmath.exp(sgn * 1j * a * math.pi)
                prod = 1
                for j in range(q):
                    prod *= lam + ph * r ** a * cmath.exp(2j * math.pi * j * a)
                rhs = lam ** q - sigma * r ** p
                worst = max(worst, abs(prod - rhs) / abs(rhs))
    print(f"(4) product identity: max relative error {worst:.1e}")


def check_final_window(alpha=0.75, lam=30.0, h=0.7, T=2.0, N=4000):
    def solve(c, f):
        dt = T / N
        t = np.arange(N + 1) * dt
        k = np.arange(N + 1)
        b = (k + 1) ** (1 - alpha) - k ** (1 - alpha)
        C = dt ** (-alpha) / gamma(2 - alpha)
        m = int(round(h / dt))
        y = np.zeros(N + 1)
        d = np.zeros(N)
        for j in range(1, N + 1):
            hist = (b[1:j] * d[j - 2::-1][:j - 1]).sum() if j > 1 else 0.0
            yd = y[j - m] if j - m >= 0 else 0.0
            y[j] = (C * y[j - 1] - C * hist - c * lam * yd + f(t[j])) / (C + lam)
            d[j - 1] = y[j] - y[j - 1]
        return y[-1]
    win = lambda t: np.sin(5 * t) * (t > T - 0.5)
    vals = [solve(c, win) for c in (0.0, 0.5, -2.0)]
    print(f"(5) final-window control, y(T) for c=0, 0.5, -2: "
          + ", ".join(f"{v:+.6e}" for v in vals))


if __name__ == "__main__":
    check_chain()
    check_duality()
    check_bernstein()
    check_product()
    check_final_window()
