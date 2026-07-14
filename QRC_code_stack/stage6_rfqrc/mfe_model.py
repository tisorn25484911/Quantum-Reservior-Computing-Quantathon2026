"""mfe_model.py -- Moehlis-Faisst-Eckhardt nine-mode shear-flow model.

Equations transcribed VERBATIM from the original paper, eqs. (21)-(32) of
Moehlis, Faisst & Eckhardt, "A low-dimensional model for turbulent shear
flows", New J. Phys. 6, 56 (2004) (author-hosted PDF verified live
2026-07-14). Configuration: the paper's free-slip domain with
Lx = 4*pi, Lz = 2*pi ("NBC" in the follow-up literature), i.e.

    alpha = 2*pi/Lx = 0.5,  beta = pi/2,  gamma = 2*pi/Lz = 1.0.

Study parameters follow Ahmed, Novoa, Dalton & Magri, Phys. Rev. Research
6, 043082 (2024) (arXiv:2405.03390, verified live 2026-07-14): Re = 400,
dt = 0.25, leading Lyapunov exponent Lambda ~ 0.0163 so one Lyapunov time
LT ~ 61.3 time units; kinetic energy k(t) = (1/2) sum_i a_i^2; extreme
event k >= k_e = 0.1; laminarised series discarded when k > k_l = 0.48.

Two structural anchors the paper supplies (both in run_anchors(), also in
stage0_anchors/test_rfqrc_anchors.py):
  1. the laminar state a = (1, 0, ..., 0) is an exact fixed point;
  2. the quadratic (advective) terms conserve energy exactly:
     sum_i a_i * N_i(a) = 0 for every a -- a typo in ANY nonlinear
     coefficient breaks this identity generically.
"""

from __future__ import annotations

import numpy as np

# NBC domain (Lx = 4 pi, Lz = 2 pi)
ALPHA = 0.5
BETA = np.pi / 2.0
GAMMA = 1.0

# Ahmed et al. 2024 study parameters
RE_DEFAULT = 400.0
DT_DEFAULT = 0.25
LYAP = 0.0163                      # leading Lyapunov exponent at Re=400
LT = 1.0 / LYAP                    # one Lyapunov time ~ 61.3 time units
K_EXTREME = 0.1                    # extreme-event threshold on k(t)
K_LAMINAR = 0.48                   # laminarisation threshold on k(t)

_KAG = np.sqrt(ALPHA ** 2 + GAMMA ** 2)
_KBG = np.sqrt(BETA ** 2 + GAMMA ** 2)
_KABG = np.sqrt(ALPHA ** 2 + BETA ** 2 + GAMMA ** 2)
_S23 = np.sqrt(2.0 / 3.0)
_S32 = np.sqrt(3.0 / 2.0)
_S6 = np.sqrt(6.0)


def rhs(a: np.ndarray, Re: float = RE_DEFAULT) -> np.ndarray:
    """da/dt for the nine-mode model. `a` has shape (9,); 0-indexed here,
    a[0] = a_1 of the paper."""
    al, be, ga = ALPHA, BETA, GAMMA
    kag, kbg, kabg = _KAG, _KBG, _KABG
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = a
    d = np.empty(9)

    # eq. (21)
    d[0] = (be ** 2 / Re
            - (be ** 2 / Re) * a1
            - _S32 * be * ga / kabg * a6 * a8
            + _S32 * be * ga / kbg * a2 * a3)
    # eq. (22)
    d[1] = (-(4.0 * be ** 2 / 3.0 + ga ** 2) * a2 / Re
            + (5.0 / 3.0) * _S23 * ga ** 2 / kag * a4 * a6
            - ga ** 2 / (_S6 * kag) * a5 * a7
            - al * be * ga / (_S6 * kag * kabg) * a5 * a8
            - _S32 * be * ga / kbg * a1 * a3
            - _S32 * be * ga / kbg * a3 * a9)
    # eq. (23)
    d[2] = (-(be ** 2 + ga ** 2) / Re * a3
            + 2.0 / _S6 * al * be * ga / (kag * kbg) * (a4 * a7 + a5 * a6)
            + (be ** 2 * (3.0 * al ** 2 + ga ** 2)
               - 3.0 * ga ** 2 * (al ** 2 + ga ** 2))
            / (_S6 * kag * kbg * kabg) * a4 * a8)
    # eq. (24)
    d[3] = (-(3.0 * al ** 2 + 4.0 * be ** 2) / (3.0 * Re) * a4
            - al / _S6 * a1 * a5
            - (10.0 / (3.0 * _S6)) * al ** 2 / kag * a2 * a6
            - _S32 * al * be * ga / (kag * kbg) * a3 * a7
            - _S32 * al ** 2 * be ** 2 / (kag * kbg * kabg) * a3 * a8
            - al / _S6 * a5 * a9)
    # eq. (25)
    d[4] = (-(al ** 2 + be ** 2) / Re * a5
            + al / _S6 * a1 * a4
            + al ** 2 / (_S6 * kag) * a2 * a7
            - al * be * ga / (_S6 * kag * kabg) * a2 * a8
            + al / _S6 * a4 * a9
            + 2.0 / _S6 * al * be * ga / (kag * kbg) * a3 * a6)
    # eq. (26)
    d[5] = (-(3.0 * al ** 2 + 4.0 * be ** 2 + 3.0 * ga ** 2) / (3.0 * Re) * a6
            + al / _S6 * a1 * a7
            + _S32 * be * ga / kabg * a1 * a8
            + (10.0 / (3.0 * _S6)) * (al ** 2 - ga ** 2) / kag * a2 * a4
            - 2.0 * _S23 * al * be * ga / (kag * kbg) * a3 * a5
            + al / _S6 * a7 * a9
            + _S32 * be * ga / kabg * a8 * a9)
    # eq. (27)
    d[6] = (-(al ** 2 + be ** 2 + ga ** 2) / Re * a7
            - al / _S6 * (a1 * a6 + a6 * a9)
            + (ga ** 2 - al ** 2) / (_S6 * kag) * a2 * a5
            + al * be * ga / (_S6 * kag * kbg) * a3 * a4)
    # eq. (28)
    d[7] = (-(al ** 2 + be ** 2 + ga ** 2) / Re * a8
            + 2.0 / _S6 * al * be * ga / (kag * kabg) * a2 * a5
            + ga ** 2 * (3.0 * al ** 2 - be ** 2 + 3.0 * ga ** 2)
            / (_S6 * kag * kbg * kabg) * a3 * a4)
    # eq. (29)
    d[8] = (-9.0 * be ** 2 / Re * a9
            + _S32 * be * ga / kbg * a2 * a3
            - _S32 * be * ga / kabg * a6 * a8)
    return d


def nonlinear_part(a: np.ndarray) -> np.ndarray:
    """Quadratic terms only (used by the energy-conservation anchor):
    N(a) = rhs(a) - linear/forcing part, extracted as
    rhs(a; Re) - [rhs at same a with quadratic terms removed]. Computed as
    rhs(a) - rhs_linear(a) where rhs_linear uses the exact linear+forcing
    structure (the linear terms all scale as 1/Re, quadratics don't --
    but rather than exploit scaling we recompute explicitly)."""
    Re = RE_DEFAULT
    full = rhs(a, Re)
    lin = np.empty(9)
    be = BETA
    lam = np.array([
        be ** 2,
        4.0 * be ** 2 / 3.0 + GAMMA ** 2,
        be ** 2 + GAMMA ** 2,
        (3.0 * ALPHA ** 2 + 4.0 * be ** 2) / 3.0,
        ALPHA ** 2 + be ** 2,
        (3.0 * ALPHA ** 2 + 4.0 * be ** 2 + 3.0 * GAMMA ** 2) / 3.0,
        ALPHA ** 2 + be ** 2 + GAMMA ** 2,
        ALPHA ** 2 + be ** 2 + GAMMA ** 2,
        9.0 * be ** 2,
    ])
    lin = -lam * a / Re
    lin[0] += be ** 2 / Re          # constant forcing in the a_1 equation
    return full - lin


def rk4_step(a: np.ndarray, dt: float, Re: float = RE_DEFAULT) -> np.ndarray:
    k1 = rhs(a, Re)
    k2 = rhs(a + 0.5 * dt * k1, Re)
    k3 = rhs(a + 0.5 * dt * k2, Re)
    k4 = rhs(a + dt * k3, Re)
    return a + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(a0: np.ndarray, n_steps: int, dt: float = DT_DEFAULT,
              Re: float = RE_DEFAULT) -> np.ndarray:
    """Trajectory of shape (n_steps + 1, 9) including the initial state."""
    traj = np.empty((n_steps + 1, 9))
    traj[0] = a0
    for i in range(n_steps):
        traj[i + 1] = rk4_step(traj[i], dt, Re)
    return traj


def kinetic_energy(traj: np.ndarray) -> np.ndarray:
    """k(t) = (1/2) sum_i a_i^2 (Ahmed et al. convention; note the
    laminar state has k = 0.5, and the turbulent attractor lives at
    smaller k)."""
    return 0.5 * np.sum(np.asarray(traj) ** 2, axis=-1)


def initial_condition(rng: np.random.Generator,
                      scale: float = 0.1) -> np.ndarray:
    """Perturbation around the (unstable-to-finite-perturbations)
    turbulence-adjacent region: laminar profile amplitude reduced, small
    random components elsewhere."""
    a0 = rng.normal(0.0, scale, size=9)
    a0[0] = 1.0 + rng.normal(0.0, scale)
    return a0


def generate_ensemble(n_series: int, n_steps: int, dt: float = DT_DEFAULT,
                      Re: float = RE_DEFAULT, seed: int = 7,
                      discard_laminarised: bool = True,
                      washout_steps: int = 400):
    """Ensemble of k(t)-filtered trajectories.

    Returns (list of trajectories with the transient removed, number
    discarded). A series is discarded when k(t) exceeds K_LAMINAR anywhere
    after the transient - it has (re)laminarised (Ahmed et al. discard
    protocol; they retain ~73% at these parameters).
    """
    rng = np.random.default_rng(seed)
    kept, discarded = [], 0
    while len(kept) < n_series:
        a0 = initial_condition(rng)
        traj = integrate(a0, n_steps + washout_steps, dt, Re)[washout_steps:]
        if discard_laminarised and np.any(kinetic_energy(traj) > K_LAMINAR):
            discarded += 1
            # give up eventually if parameters make survival too rare
            if discarded > 50 * max(n_series, 1):
                raise RuntimeError("laminarisation filter rejecting "
                                   ">98% of series; check parameters")
            continue
        kept.append(traj)
    return kept, discarded


# ------------------------------------------------------------------ anchors
def run_anchors(verbose: bool = True) -> bool:
    ok = True

    # 1. laminar fixed point is exactly stationary
    a_lam = np.zeros(9)
    a_lam[0] = 1.0
    r = np.max(np.abs(rhs(a_lam)))
    good = r < 1e-15
    ok &= good
    if verbose:
        print(f"  [{'PASS' if good else 'FAIL'}] laminar state stationary: "
              f"max|da/dt| = {r:.2e}")

    # 2. quadratic terms conserve energy: a . N(a) = 0 for random a
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20):
        a = rng.normal(size=9)
        worst = max(worst, abs(float(a @ nonlinear_part(a))))
    good = worst < 1e-12
    ok &= good
    if verbose:
        print(f"  [{'PASS' if good else 'FAIL'}] nonlinear energy "
              f"conservation: max|a.N(a)| = {worst:.2e}")

    # 3. integrator sanity: short trajectory stays finite, k in range
    traj = integrate(initial_condition(np.random.default_rng(1)), 2000)
    k = kinetic_energy(traj)
    good = np.all(np.isfinite(traj)) and 0 < k.mean() < 1.0
    ok &= good
    if verbose:
        print(f"  [{'PASS' if good else 'FAIL'}] 2000-step trajectory "
              f"finite; mean k = {k.mean():.3f}")
    return ok


if __name__ == "__main__":
    import sys
    print("mfe_model anchors:")
    sys.exit(0 if run_anchors() else 1)
