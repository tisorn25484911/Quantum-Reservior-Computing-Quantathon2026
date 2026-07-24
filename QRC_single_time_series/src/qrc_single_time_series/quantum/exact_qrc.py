"""exact_qrc.py -- exact NumPy FN reservoir (density matrix, virtual nodes).

The paper-faithful Fujii-Nakajima reservoir, evolved exactly as a density matrix
(no shots, no Trotter) -- the ceiling every other implementation is measured
against. Per step k the injection qubit (site 0, G2) is re-prepared in the
input-dependent state rho_s(u_k) (CPTP, spec s8.3), then the register evolves for
one interval tau, read out at V virtual-node sub-times

    t = k*tau + (v+1)*tau/V,   v = 0..V-1                    (errata G8-4)

giving features x'_{n,v} = (1 + <O_n(t)>)/2 plus a bias column.

Efficiency (phase file): with sigma = W^dag rho W in the energy eigenbasis
(H = W diag(lambda) W^dag), rho(t) = W (sigma . Phi(t)) W^dag where
Phi(t)_{ab} = exp(-i (lambda_a - lambda_b) t). All V readouts at a step are one
einsum against a precomputed phase stack; advancing the state a full tau is one
basis round trip rho <- W (sigma . Phi(tau)) W^dag. Implemented in P2.
"""
from __future__ import annotations

import numpy as np

from .hamiltonians import Reservoir, fc_tfi
from .input_channels import inject
from .observables import build_observables


class ExactQRC:
    """Exact density-matrix FN reservoir with V virtual nodes.

    Parameters
    ----------
    reservoir : Reservoir -- frozen Hamiltonian (see ``hamiltonians``).
    V : int -- virtual nodes per input step.
    tau : float -- physical evolution time per input step.
    observable_kind : str -- passed to ``build_observables`` ("z_local" default).
    inject_site : int -- injection qubit (G2 default 0).
    """

    def __init__(self, reservoir, V=10, tau=2.0, observable_kind="z_local",
                 observable_strings=None, inject_site=0):
        self.res = reservoir
        self.N = reservoir.N
        self.V = int(V)
        self.tau = float(tau)
        self.inject_site = inject_site
        self.ops, self.obs_labels = build_observables(
            self.N, observable_kind, observable_strings)
        self.M = len(self.ops)

        w, W = reservoir.eig()
        self.w, self.W, self.Wd = w, W, W.conj().T
        omega = w[:, None] - w[None, :]                      # (d, d)
        # virtual-node sub-times t_v = (v+1) * tau / V, v = 0..V-1
        self._sub_times = (np.arange(1, self.V + 1) * self.tau / self.V)
        self._Phi_sub = np.exp(-1j * omega[None, :, :] * self._sub_times[:, None, None])
        self._Phi_end = np.exp(-1j * omega * self.tau)       # advance a full tau
        # observables in the energy eigenbasis, transposed for the einsum contraction
        self._ObsT = np.stack([(self.Wd @ op @ self.W).T for op in self.ops])  # (M,d,d)

    # -- construction helpers -------------------------------------------------
    @classmethod
    def from_config(cls, cfg):
        """Build from a ``qrc_exact.yaml``-style dict."""
        ham = fc_tfi(cfg["n_qubits"], J=cfg.get("J", 1.0), h=cfg.get("h", 1.0),
                     seed=cfg.get("reservoir_seed", 7))
        return cls(ham, V=cfg["V_virtual_nodes"], tau=cfg["tau"],
                   observable_kind=cfg.get("observables", "z_local"))

    def initial_state(self):
        """|0...0><0...0| density matrix (default reservoir start)."""
        dim = 2 ** self.N
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 0] = 1.0
        return rho

    @property
    def feature_dim(self):
        """M * V + 1 (bias)."""
        return self.M * self.V + 1

    # -- expectations ---------------------------------------------------------
    def _sigma(self, rho):
        return self.Wd @ rho @ self.W

    def _obs_at_subtimes(self, sigma):
        """<O_n(t_v)> for all observables n and sub-times v -> (V, M) real array."""
        # <O_n(t_v)> = sum_ab ObsT_n[ab] * sigma[ab] * Phi_v[ab]
        vals = np.einsum("vab,ab,mab->vm", self._Phi_sub, sigma, self._ObsT)
        return vals.real

    def _obs_postinjection(self, sigma):
        """<O_n> at the post-injection instant (v = -1, no evolution) -> (M,) real."""
        vals = np.einsum("ab,mab->m", sigma, self._ObsT)
        return vals.real

    # -- feature extraction ---------------------------------------------------
    def features(self, inputs, x0=None, sample="virtual", include_bias=True,
                 check_budget=True):
        """Feature matrix for an input sequence.

        ``sample='virtual'`` -> V virtual nodes per step, feature dim M*V (+bias).
        ``sample='postinjection'`` -> the single debug v=-1 sample per step, dim M
        (+bias); used by the G3 STM tau_B=0 exactness test.
        """
        s = np.asarray(inputs, dtype=float)
        L = len(s)
        per_step = self.M * (self.V if sample == "virtual" else 1)
        if check_budget:
            fc = per_step + (1 if include_bias else 0)
            if fc > L / 5:
                raise ValueError(
                    f"feature count {fc} exceeds L/5 = {L/5:.1f} (FN overdetermination "
                    f"assumption); lengthen L or cap V/observables")
        rho = self.initial_state() if x0 is None else np.array(x0, dtype=complex)
        rows = np.empty((L, per_step), dtype=float)
        for k in range(L):
            rho = inject(rho, float(s[k]), self.N, site=self.inject_site)
            sigma = self._sigma(rho)
            if sample == "virtual":
                obs = self._obs_at_subtimes(sigma)          # (V, M)
                rows[k] = (0.5 * (1.0 + obs)).reshape(-1)   # x' = (1+<O>)/2
                # advance a full tau for the next injection
                rho = self.W @ (sigma * self._Phi_end) @ self.Wd
            else:  # postinjection: read v=-1, still advance one tau
                obs = self._obs_postinjection(sigma)        # (M,)
                rows[k] = 0.5 * (1.0 + obs)
                rho = self.W @ (sigma * self._Phi_end) @ self.Wd
        if include_bias:
            rows = np.hstack([rows, np.ones((L, 1))])
        return rows

    # -- reference generator (materialises rho; slow path for exactness tests) -
    def run(self, inputs, x0=None, debug_postinjection=False):
        """Yield ``(k, v, rho)`` at every virtual-node time (reference/debug path).

        ``v`` runs 0..V-1; with ``debug_postinjection`` a v=-1 tuple (the state
        immediately after injection, before evolution) is yielded first per step.
        The state carried between steps is rho(k*tau + tau) = rho after V sub-steps.
        """
        s = np.asarray(inputs, dtype=float)
        rho = self.initial_state() if x0 is None else np.array(x0, dtype=complex)
        for k in range(len(s)):
            rho = inject(rho, float(s[k]), self.N, site=self.inject_site)
            sigma = self._sigma(rho)
            if debug_postinjection:
                yield k, -1, rho.copy()
            for v in range(self.V):
                rho_tv = self.W @ (sigma * self._Phi_sub[v]) @ self.Wd
                yield k, v, rho_tv
            rho = self.W @ (sigma * self._Phi_end) @ self.Wd
