# Controllability of subdiffusion with a delay in the diffusion term — numerical experiments

This repository contains the code for the numerical experiments and the
numerical checks in

> B. Sikora, *Rest, final state and delay state: controllability of
> subdiffusion equations with a delay in the diffusion term*,
> submitted to SIAM Journal on Control and Optimization (2026).

The paper studies

$$
\partial_t^\alpha y - \Delta y - c\,\Delta y(\cdot-h) = \mathbf 1_\omega u
\quad\text{in } \Omega\times(0,T),
$$

with homogeneous Dirichlet boundary conditions, initial state $y_0$ and
initial history $\varphi$ on $[-h,0)$, where $\partial_t^\alpha$ is the
Caputo derivative of order $\alpha\in(\tfrac12,1)$, $h>0$ is the delay,
$c\in\mathbb R$, $y$ is the state and $u$ is the control acting in
$\omega\subset\Omega$.

## Method

* **Space:** spectral Galerkin method with the eigenfunctions
  $e_n(x)=\sqrt2\sin(n\pi x)$ of $-\partial_{xx}$ on $\Omega=(0,1)$.
* **Time:** L1 scheme for the Caputo derivative on a uniform grid, implicit
  in $-\Delta y(t)$. The delay $h$ is a multiple of the time step, so the
  delayed term is known from earlier time levels (or from the history) and is
  treated explicitly. No Mittag-Leffler functions are evaluated.
* **Controls (Section 8.3):** controls supported in the final window
  $(T-\varepsilon,T)$, $\varepsilon<h$, built from the Gramian of the problem
  without delay on $[0,\varepsilon]$ (Lemma 6.1, Remark 6.3(b) of the paper).
* **Characteristic roots (Section 8.1):** zeros of
  $\Delta_n(s)=s^\alpha+\lambda_n(1+c\,e^{-sh})$ computed by Newton's method
  (mpmath) from the first-order approximation of Remark 3.6(a).

## Installation

```bash
git clone https://github.com/BeSikora/subdiffusion-delay-control.git
cd subdiffusion-delay-control
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Repository layout

```
.
├── delay_experiments.py   # Figures 1-3, Tables 3-4 of the paper
├── checks.py              # numerical checks quoted in the text
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

All output files are written to the current working directory.

## Reproducing the results

```bash
python delay_experiments.py      # all figures and tables
python checks.py                 # all numerical checks
```

Individual items (numbering as in the submitted version of the paper):

| Item in the paper | Content | Command | Output |
|---|---|---|---|
| Figure 1 | zeros of $\Delta_n$ and the neutral chain, $c\in\{0.5,1,2\}$ | `python delay_experiments.py --roots` | `fig_roots.pdf` |
| Figure 2, Sec. 8.2 | transport of the history, $c\in\{0.5,-2\}$ | `python delay_experiments.py --transport` | `fig_transport.pdf`, errors on screen |
| Figure 3, Table 4 | final-window controls, $c\in\{0,0.5,2\}$ | `python delay_experiments.py --control` | `fig_delay_control.pdf`, values on screen |
| Table 3 | grid convergence for $c=2$ | `python delay_experiments.py --convergence` | `table_delay_convergence.tex` |
| Remark 3.6(a) | accuracy of the first-order formula for the zeros | `python checks.py` | screen |
| Lemma 5.3 | duality identity (single mode) | `python checks.py` | screen |
| Lemma 5.7 | Bernstein representation of the kernel | `python checks.py` | screen |
| Proposition 5.9 | product identity for rational $\alpha=p/q$ | `python checks.py` | screen |
| Lemma 6.1 | final-window controls do not see the delay | `python checks.py` | screen |

Each command runs in a few seconds on a standard laptop. Figures are saved as
PDF with TrueType fonts embedded.

### Checking the output

`python delay_experiments.py --control` should print, among others,

| $c$ | $\lVert y^{\mathrm{free}}(T)\rVert$ | $\delta=10^{-6}$: error | $\delta=10^{-6}$: cost |
|---|---|---|---|
| 0   | 0.022 | 4.97e-02 | 2.94e+01 |
| 0.5 | 0.145 | 5.00e-02 | 2.77e+01 |
| 2   | 2.375 | 5.62e-02 | 2.60e+01 |

and a maximal identity defect $\lVert y_\delta(T)-(y_1-\delta R^0(\delta)p)\rVert$
of about $3\cdot10^{-10}$ for all three values of $c$. `python checks.py`
should report errors of order $10^{-9}$ (1), $10^{-6}$ (2), below
$10^{-20}$ (3), $10^{-14}$ (4), and three identical values in (5).

All numbers quoted in Section 8 and in the numerical remarks of the paper were
reproduced exactly with the versions listed in `requirements.txt`. For
$\delta=10^{-10}$ the cost is not converged with respect to the time grid
(Table 3), so small differences in the last digits may occur with other
versions of the linear algebra libraries.

## How to cite

GitHub shows the citation in the sidebar ("Cite this repository"), based on
`CITATION.cff`.

```bibtex
@article{Sikora2026delay,
  author  = {Sikora, Beata},
  title   = {Rest, final state and delay state: controllability of
             subdiffusion equations with a delay in the diffusion term},
  journal = {SIAM Journal on Control and Optimization},
  year    = {2026},
  note    = {Submitted}
}
```

## License

MIT License; see [`LICENSE`](LICENSE).
