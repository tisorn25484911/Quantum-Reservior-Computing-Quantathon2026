"""statistical.py -- ARIMA / SARIMA baselines via statsmodels (AIC + dev selection).

Classical statistical forecasters. Orders are chosen by AIC over a small grid and
the winner is re-scored by ONE-STEP rolling forecast on the DEV span only (never
test). The fitted model exposes:

  - ``predict_onestep`` : in-sample one-step-ahead predictions (teacher-forced
    table; statsmodels ``get_prediction``), and
  - ``warm``/``step``   : a closed-loop adapter that appends each fed-back value and
    reads the next one-step forecast, so ARIMA recursion flows through the SAME
    autonomous engine as every other model (spec s24, G5).

Recursive multi-step ARIMA forecasting IS feed-your-own-prediction-back by
construction; the adapter makes that explicit and comparable. Implemented in P7.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:                                                      # statsmodels is a core dep
    from statsmodels.tsa.arima.model import ARIMA
    _SM = True
except Exception:                                         # pragma: no cover
    _SM = False


@dataclass
class ARIMAModel:
    order: tuple
    seasonal_order: tuple
    result: object                # fitted statsmodels result
    endog: np.ndarray             # training endog (for the rolling adapter)
    aic: float

    def predict_onestep(self):
        """In-sample one-step-ahead predictions over the fitted span."""
        pr = self.result.get_prediction()
        return np.asarray(pr.predicted_mean, dtype=float)

    # -- autonomous adapter --------------------------------------------------
    def warm(self, history):
        """Refit-free state: append the observed history to the fitted model."""
        h = np.asarray(history, dtype=float)
        res = self.result.apply(h[:-1], refit=False) if len(h) > 1 else self.result
        return {"res": res}

    def step(self, state, u):
        res = state["res"].append([u], refit=False)
        yhat = float(np.asarray(res.forecast(1), dtype=float)[0])
        return {"res": res}, yhat


def _fit_order(endog, order, seasonal_order):
    kw = dict(order=order, enforce_stationarity=False, enforce_invertibility=False)
    if seasonal_order is not None:
        kw["seasonal_order"] = seasonal_order
    return ARIMA(endog, **kw).fit()


def fit(y, n_train, orders=None, seasonal_orders=(None,)):
    """Select (order[, seasonal_order]) by AIC on the first ``n_train`` points.

    ``orders`` default is a small non-seasonal ARIMA grid; pass seasonal orders
    (e.g. ``[(1,0,0,12)]``) for SARIMA on monthly data.
    """
    if not _SM:                                           # pragma: no cover
        raise RuntimeError("statistical.py requires statsmodels")
    endog = np.asarray(y, dtype=float)[:n_train]
    if orders is None:
        orders = [(1, 0, 0), (2, 0, 0), (1, 0, 1), (2, 0, 1), (1, 1, 1)]
    best, best_aic = None, np.inf
    for od in orders:
        for so in seasonal_orders:
            try:
                res = _fit_order(endog, od, so)
            except Exception:                             # pragma: no cover
                continue
            if res.aic < best_aic:
                best_aic, best = res.aic, (od, so, res)
    if best is None:                                      # pragma: no cover
        raise RuntimeError("no ARIMA order fit successfully")
    od, so, res = best
    return ARIMAModel(order=od, seasonal_order=so, result=res,
                      endog=endog, aic=float(best_aic))
