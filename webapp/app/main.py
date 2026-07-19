"""main.py -- FastAPI application: routes, JSON API, and chart endpoints.

Two surfaces over one domain layer:

* **HTML** (Jinja + HTMX) for the demo itself. HTMX handles progress polling
  and partial swaps, so there is no build step and no client-side framework.
* **JSON** under ``/api`` for anything programmatic, documented automatically
  at ``/docs``.

Both call the same functions in ``app.domain``; no computation lives in a route.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import (CLAIM_DISCLAIMER, SEED, SEED as PROJECT_SEED,
                     STATIC_DIR, TEMPLATES_DIR, TIER_LABELS)
from .domain import catalog, charts
from .domain.engine import (ForecastConfig, run_forecast, run_horizon_sweep,
                            run_memory_capacity)
from .domain.jobs import registry

app = FastAPI(
    title="QRC Forecasting Demo",
    description=("Quantum reservoir computing applied to time-series "
                 "forecasting. " + CLAIM_DISCLAIMER),
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals["disclaimer"] = CLAIM_DISCLAIMER
templates.env.globals["seed"] = PROJECT_SEED


def _fmt(value, spec=".4g"):
    """Jinja filter: format a number, or render a dash when it is missing."""
    if value is None:
        return "--"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return "--"
    return format(f, spec)


templates.env.filters["num"] = _fmt
templates.env.filters["pct"] = lambda v: ("--" if v is None
                                          or not np.isfinite(float(v))
                                          else f"{float(v) * 100:.1f}%")


def _cfg_from_form(**kw) -> ForecastConfig:
    """Build a validated config from form fields, as a 400 rather than a 500."""
    try:
        cfg = ForecastConfig(**kw)
        cfg.validate()
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return cfg


def _require_result(job_id: str, kind: str | None = None) -> dict:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no such run: {job_id}")
    if job.status == "error":
        raise HTTPException(status_code=409,
                            detail=f"run failed: {job.error}")
    if job.result is None:
        raise HTTPException(status_code=409, detail="run is not finished yet")
    if kind and job.kind != kind:
        raise HTTPException(status_code=409,
                            detail=f"run {job_id} is a {job.kind} run")
    return job.result


# ======================================================================
# HTML pages
# ======================================================================
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    datasets = catalog.list_datasets()
    featured = [d for d in datasets if d.get("featured")]
    return templates.TemplateResponse(request, "index.html", {
        "featured": featured,
        "n_datasets": len(datasets),
        "recent": registry.list(limit=5),
        "defaults": asdict(ForecastConfig()),
    })


@app.get("/datasets", response_class=HTMLResponse)
def datasets_page(request: Request, tier: str | None = None):
    return templates.TemplateResponse(request, "datasets.html", {
        "datasets": catalog.list_datasets(tier),
        "tier": tier, "tiers": TIER_LABELS,
    })


@app.get("/datasets/{key}", response_class=HTMLResponse)
def dataset_detail(request: Request, key: str):
    try:
        meta = catalog.describe(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return templates.TemplateResponse(request, "dataset_detail.html", {
        "d": meta, "defaults": asdict(ForecastConfig()),
    })


@app.get("/forecast", response_class=HTMLResponse)
def forecast_page(request: Request, dataset: str = "solar"):
    return templates.TemplateResponse(request, "forecast.html", {
        "datasets": catalog.list_datasets(),
        "defaults": {**asdict(ForecastConfig()), "dataset": dataset},
    })


@app.post("/forecast", response_class=HTMLResponse)
def forecast_submit(
    request: Request,
    dataset: str = Form("solar"),
    horizon: int = Form(1),
    n_qubits: int = Form(5),
    dt: float = Form(2.0),
    virtual_nodes: int = Form(4),
    use_zz: bool = Form(False),
    alpha: float = Form(0.1),
    max_points: int = Form(2000),
    seed: int = Form(SEED),
    mode: str = Form("single"),
):
    cfg = _cfg_from_form(dataset=dataset, horizon=horizon, n_qubits=n_qubits,
                         dt=dt, virtual_nodes=virtual_nodes, use_zz=use_zz,
                         alpha=alpha, max_points=max_points, seed=seed)
    if mode == "sweep":
        job = registry.submit("sweep", f"{dataset} horizon sweep",
                              run_horizon_sweep, cfg)
    else:
        job = registry.submit("forecast", f"{dataset} h={horizon}",
                              run_forecast, cfg)
    return RedirectResponse(f"/runs/{job.id}", status_code=303)


@app.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request):
    return templates.TemplateResponse(request, "runs.html", {
        "jobs": registry.list(limit=50)})


@app.get("/runs/{job_id}", response_class=HTMLResponse)
def run_detail(request: Request, job_id: str):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no such run: {job_id}")
    tpl = {"forecast": "run_forecast.html", "sweep": "run_sweep.html",
           "memory": "run_memory.html"}.get(job.kind, "run_forecast.html")
    if job.status != "done":
        tpl = "run_pending.html"
    return templates.TemplateResponse(request, tpl, {
        "job": job.public(), "r": job.result})


@app.get("/runs/{job_id}/progress", response_class=HTMLResponse)
def run_progress(request: Request, job_id: str):
    """HTMX partial: the progress bar, polled while a job is running."""
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such run")
    return templates.TemplateResponse(request, "_progress.html", {
        "job": job.public()})


@app.get("/diagnostics", response_class=HTMLResponse)
def diagnostics_page(request: Request):
    return templates.TemplateResponse(request, "diagnostics.html", {
        "defaults": asdict(ForecastConfig())})


@app.post("/diagnostics", response_class=HTMLResponse)
def diagnostics_submit(
    n_qubits: int = Form(5),
    dt: float = Form(2.0),
    virtual_nodes: int = Form(4),
    use_zz: bool = Form(False),
    seed: int = Form(SEED),
    max_delay: int = Form(25),
):
    cfg = _cfg_from_form(n_qubits=n_qubits, dt=dt, virtual_nodes=virtual_nodes,
                         use_zz=use_zz, seed=seed)
    job = registry.submit("memory", f"memory capacity, {n_qubits}q",
                          run_memory_capacity, cfg, max_delay=int(max_delay))
    return RedirectResponse(f"/runs/{job.id}", status_code=303)


@app.get("/about", response_class=HTMLResponse)
def about_page(request: Request):
    return templates.TemplateResponse(request, "about.html", {})


# ======================================================================
# Charts
# ======================================================================
def _png_response(data: bytes) -> Response:
    return Response(content=data, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/charts/series/{key}.png")
def chart_series(key: str):
    try:
        x, meta = catalog.series_values(key, max_points=4000)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _png_response(charts.series_png(x, meta))


@app.get("/charts/forecast/{job_id}.png")
def chart_forecast(job_id: str, n_show: int = 240):
    return _png_response(charts.forecast_png(_require_result(job_id),
                                             n_show=n_show))


@app.get("/charts/scatter/{job_id}.png")
def chart_scatter(job_id: str):
    return _png_response(charts.scatter_png(_require_result(job_id)))


@app.get("/charts/horizon/{job_id}.png")
def chart_horizon(job_id: str):
    return _png_response(charts.horizon_png(_require_result(job_id, "sweep")))


@app.get("/charts/memory/{job_id}.png")
def chart_memory(job_id: str):
    return _png_response(charts.memory_png(_require_result(job_id, "memory")))


# ======================================================================
# JSON API
# ======================================================================
@app.get("/api/health")
def api_health():
    return {"status": "ok", "seed": PROJECT_SEED,
            "n_datasets": len(catalog.list_datasets()),
            "disclaimer": CLAIM_DISCLAIMER}


@app.get("/api/datasets")
def api_datasets(tier: str | None = None):
    return {"datasets": catalog.list_datasets(tier)}


@app.get("/api/datasets/{key}")
def api_dataset(key: str):
    try:
        return catalog.describe(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/runs")
def api_submit(payload: dict):
    kind = payload.pop("kind", "forecast")
    extra = {}
    if kind == "sweep":
        extra["horizons"] = payload.pop("horizons", None)
    if kind == "memory":
        extra["max_delay"] = int(payload.pop("max_delay", 25))
    cfg = _cfg_from_form(**payload)
    fn = {"forecast": run_forecast, "sweep": run_horizon_sweep,
          "memory": run_memory_capacity}.get(kind)
    if fn is None:
        raise HTTPException(status_code=400, detail=f"unknown kind: {kind}")
    job = registry.submit(kind, f"{kind}: {cfg.dataset}", fn, cfg, **extra)
    return JSONResponse(job.public(), status_code=202)


@app.get("/api/runs")
def api_runs(kind: str | None = None, limit: int = 50):
    return {"runs": registry.list(kind, limit)}


@app.get("/api/runs/{job_id}")
def api_run(job_id: str):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no such run: {job_id}")
    return job.public(include_result=job.status == "done")
