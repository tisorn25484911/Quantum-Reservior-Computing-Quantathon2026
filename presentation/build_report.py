"""build_report.py -- assemble the shareable results report (self-contained HTML).

Embeds the real result figures as data URIs and the Step-1 skill numbers as an
inline SVG, so the output is one file with no external assets. Rebuild with:

    python build_report.py    # -> results_report.html
"""
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
AF = HERE.parent / "Quantathon_stack" / "Anomaly_Forecast"


def datauri(p):
    b = Path(p).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()


# --- Step 1 anomaly-target numbers for the inline SVG ---
ss = json.loads((AF / "results" / "step2_seedsweep_got_sst_mhwi.json").read_text())
qrc = ss["mean"]["xxz_hx"]
esn = ss["mean"]["esn"]
per = ss["baselines"]["persistence"]

FIG_MHW = datauri(AF / "labels" / "mhw_labels.png")
FIG_COMPOSE = datauri(AF / "results" / "compose_mhw_xxz_hx.png")
FIG_STEP3 = datauri(AF / "results" / "step3_stochastic_xxz_hx.png")


def svg_skill():
    """Inline SVG: NMSE vs lead for QRC / ESN / persistence (Step 1)."""
    W, H = 640, 340
    ml, mr, mt, mb = 46, 16, 14, 42
    x0, x1, y0, y1 = ml, W - mr, H - mb, mt
    xmax, ymax = 24, 1.25

    def X(h): return x0 + (h - 1) / (xmax - 1) * (x1 - x0)
    def Y(v): return y0 + (v - 0) / (ymax - 0) * (y1 - y0)

    def path(a, dash=""):
        d = " ".join(("M" if i == 0 else "L") + f"{X(i+1):.1f} {Y(v):.1f}"
                     for i, v in enumerate(a))
        return d

    g = []
    for v in (0, .25, .5, .75, 1.0, 1.25):
        g.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(v):.0f}" y2="{Y(v):.0f}" '
                 f'stroke="#d4e2e0" stroke-width="1"/>')
        g.append(f'<text x="{x0-8}" y="{Y(v)+3:.0f}" text-anchor="end" '
                 f'font-size="10" fill="#6f8a8c" font-family="monospace">{v:.2f}</text>')
    g.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(1):.0f}" y2="{Y(1):.0f}" '
             f'stroke="#6f8a8c" stroke-width="1.3" stroke-dasharray="2 4"/>')
    g.append(f'<text x="{x1-3}" y="{Y(1)-5:.0f}" text-anchor="end" font-size="9.5" '
             f'fill="#6f8a8c" font-family="monospace">no skill</text>')
    for h in (1, 6, 12, 18, 24):
        g.append(f'<text x="{X(h):.0f}" y="{y0+20}" text-anchor="middle" '
                 f'font-size="10" fill="#6f8a8c" font-family="monospace">{h}</text>')
    g.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-4}" text-anchor="middle" '
             f'font-size="10.5" fill="#6f8a8c">lead time - days ahead</text>')
    g.append(f'<path d="{path(per)}" fill="none" stroke="#6f8a8c" stroke-width="2" '
             f'stroke-dasharray="3 4"/>')
    g.append(f'<path d="{path(esn)}" fill="none" stroke="#d97a2b" stroke-width="2.3"/>')
    g.append(f'<path d="{path(qrc)}" fill="none" stroke="#1596a4" stroke-width="2.6"/>')
    # markers where persistence crosses no-skill (~h15)
    g.append(f'<circle cx="{X(24):.0f}" cy="{Y(qrc[-1]):.0f}" r="3.5" fill="#1596a4"/>')
    g.append(f'<circle cx="{X(24):.0f}" cy="{Y(per[-1]):.0f}" r="3.5" fill="#6f8a8c"/>')
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" '
            f'style="max-width:640px" role="img" aria-label="Forecast error vs '
            f'lead: QRC keeps skill to 24 days while persistence crosses no-skill '
            f'by day 15.">' + "".join(g) + '</svg>')


HTML = f"""<title>TideRead - Results Report</title>
<style>
 :root{{--bg:#f3f7f6;--surf:#fff;--surf2:#e9f1f0;--ink:#0b2a2e;--ink2:#3a5a5d;
  --ink3:#6f8a8c;--hair:#d4e2e0;--qrc:#1596a4;--esn:#d97a2b;--heat:#e0483a;--good:#2f9e6f;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;}}
 @media(prefers-color-scheme:dark){{:root{{--bg:#061417;--surf:#0c2226;--surf2:#102e33;
  --ink:#e8f3f2;--ink2:#a9c4c4;--ink3:#6f9296;--hair:#1c3b40;--qrc:#26c0d0;--esn:#e79a4d;
  --heat:#ff6b5a;--good:#4bcf95;}}}}
 :root[data-theme=dark]{{--bg:#061417;--surf:#0c2226;--surf2:#102e33;--ink:#e8f3f2;
  --ink2:#a9c4c4;--ink3:#6f9296;--hair:#1c3b40;--qrc:#26c0d0;--esn:#e79a4d;--heat:#ff6b5a;--good:#4bcf95;}}
 :root[data-theme=light]{{--bg:#f3f7f6;--surf:#fff;--surf2:#e9f1f0;--ink:#0b2a2e;--ink2:#3a5a5d;
  --ink3:#6f8a8c;--hair:#d4e2e0;--qrc:#1596a4;--esn:#d97a2b;--heat:#e0483a;--good:#2f9e6f;}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);
  font-family:var(--sans);line-height:1.6}}
 .wrap{{max-width:820px;margin:0 auto;padding:clamp(22px,5vw,60px)}}
 h1{{font-size:clamp(1.9rem,5vw,3rem);line-height:1.05;letter-spacing:-.02em;margin:0 0 .3em;text-wrap:balance}}
 h2{{font-size:1.4rem;letter-spacing:-.01em;margin:2.4em 0 .2em;padding-top:1.1em;border-top:1px solid var(--hair)}}
 h3{{font-size:1rem;margin:1.4em 0 .3em}}
 p{{margin:.6em 0;color:var(--ink2)}} strong{{color:var(--ink)}}
 .lede{{font-size:1.15rem;color:var(--ink2)}}
 .eyebrow{{font-family:var(--mono);font-size:.72rem;letter-spacing:.2em;text-transform:uppercase;color:var(--qrc);margin-bottom:1em}}
 .step{{display:flex;gap:12px;flex-wrap:wrap;margin:1em 0}}
 .step div{{flex:1 1 180px;background:var(--surf);border:1px solid var(--hair);border-radius:12px;padding:14px 16px}}
 .step .n{{font-family:var(--mono);font-size:.7rem;color:var(--qrc);letter-spacing:.1em}}
 .step .t{{font-weight:650;margin:3px 0}}
 .step .d{{font-size:.85rem;color:var(--ink3)}}
 figure{{margin:1.2em 0;background:var(--surf);border:1px solid var(--hair);border-radius:12px;padding:14px}}
 figure img{{width:100%;height:auto;border-radius:6px;background:#fff}}
 figcaption{{font-size:.82rem;color:var(--ink3);margin-top:8px}}
 .read{{background:var(--surf2);border-radius:10px;padding:13px 16px;margin:.7em 0 1.4em;font-size:.9rem}}
 .read .h{{font-family:var(--mono);font-size:.66rem;letter-spacing:.12em;text-transform:uppercase;color:var(--qrc);display:block;margin-bottom:.4em}}
 .read p{{margin:.35em 0}} .read b{{color:var(--ink)}}
 .read ul{{margin:.35em 0;padding-left:1.1em}} .read li{{color:var(--ink2);margin:.2em 0}}
 .card{{background:var(--surf);border:1px solid var(--hair);border-radius:12px;padding:16px 18px;margin:1em 0}}
 .verdict{{border-left:3px solid var(--good)}}
 .caveat{{border-left:3px solid var(--heat)}}
 table{{width:100%;border-collapse:collapse;font-size:.9rem;margin:.6em 0}}
 th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--hair)}}
 th{{font-family:var(--mono);font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;color:var(--ink3)}}
 td.n{{text-align:right;font-variant-numeric:tabular-nums;font-family:var(--mono)}}
 .win{{color:var(--good);font-weight:650}} .tie{{color:var(--ink3)}} .hard{{color:var(--heat)}}
 .legend{{font-size:.82rem;color:var(--ink2);display:flex;gap:16px;flex-wrap:wrap;margin:.4em 0}}
 .legend b{{display:inline-block;width:16px;height:3px;border-radius:2px;vertical-align:middle;margin-right:5px}}
 .toggle{{position:fixed;top:12px;right:12px;font-family:var(--mono);font-size:.68rem;
  background:var(--surf);color:var(--ink2);border:1px solid var(--hair);border-radius:99px;padding:6px 12px;cursor:pointer}}
 small{{color:var(--ink3)}}
</style>
<button class="toggle" onclick="var r=document.documentElement,d=r.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');r.setAttribute('data-theme',d=='dark'?'light':'dark')">THEME</button>
<div class="wrap">
 <div class="eyebrow">Quantathon 2026 - Quantum Reservoir Computing</div>
 <h1>Predicting marine heatwaves before they hit the farm</h1>
 <p class="lede">A quantum reservoir forecasts the ocean's temperature anomaly; a detector
  runs on that forecast to warn of a marine heatwave days ahead. Built and tested end-to-end
  on 45 years of Gulf-of-Thailand data, for CP-scale aquaculture.</p>

 <h2>How it works</h2>
 <div class="step">
  <div><div class="n">STEP 1</div><div class="t">Forecast</div><div class="d">The quantum reservoir predicts the sea-temperature anomaly, closed-loop, days ahead.</div></div>
  <div><div class="n">STEP 2</div><div class="t">Detect</div><div class="d">A detector learns real marine-heatwave events from labelled data.</div></div>
  <div><div class="n">COMPOSE</div><div class="t">Warn</div><div class="d">Run the detector on the forecast &rarr; an early warning with real lead time.</div></div>
 </div>

 <h2>The data &amp; the labels</h2>
 <p>We assembled real Gulf-of-Thailand daily sea-surface temperature (45 years), a 170-year
  reconstruction, the El Nino index, and rainfall - then labelled marine heatwaves with the
  peer-reviewed <strong>Hobday (2016)</strong> standard: <strong>165 events</strong>. Validation
  that the labels are trustworthy: the hottest heatwave we flag is the 1997-98 El Nino, and the
  strongest El Nino we detect is 2015-16 - both correct.</p>
 <figure><img src="{FIG_MHW}" alt="Gulf of Thailand SST with marine-heatwave events shaded">
  <figcaption>Gulf SST (dark) with its seasonal climatology (grey) and 90th-percentile threshold
  (orange dashed). Red = labelled marine heatwaves, only where temperature exceeds the threshold.</figcaption></figure>
 <div class="read"><span class="h">How to read this graph</span>
  <p><b>Axes:</b> horizontal = time (last 8 years); vertical = sea temperature in &deg;C.</p>
  <p><b>Dark line</b> = the actual daily temperature (note the strong yearly cycle - hot around
   April/May, cool around December). <b>Grey line</b> = the <em>normal</em> temperature for each date.
   <b>Orange dashed</b> = the heatwave threshold (unusually hot <em>for that date</em>). <b>Red</b> =
   a marine heatwave: temperature above the orange line for 5+ days.</p>
  <p><b>The point:</b> the threshold rides up and down with the seasons, so a "heatwave" means hotter
   than normal <em>for that time of year</em> - not just "summer". These red events are what the
   system learns to predict.</p></div>

 <h2>Step 1 - the forecast works, on the right target</h2>
 <p>Forecasting <em>raw</em> temperature is a trap: the ocean barely moves day-to-day, so "tomorrow
  = today" is nearly unbeatable and the quantum model only ties it. Forecasting the <strong>anomaly</strong>
  (temperature minus its seasonal normal) is where it wins - and it's the exact signal a heatwave
  detector needs.</p>
 <div class="legend"><span><b style="background:var(--qrc)"></b>Quantum reservoir</span>
  <span><b style="background:var(--esn)"></b>Classical model (same size)</span>
  <span><b style="background:var(--ink3)"></b>Persistence ("tomorrow=today")</span></div>
 <figure>{svg_skill()}
  <figcaption>Forecast error (lower better) vs lead. 1.0 = no skill. The quantum reservoir keeps
  skill to 24 days; persistence collapses past the no-skill line by ~day 15. 5-seed average.</figcaption></figure>
 <div class="read"><span class="h">How to read this graph</span>
  <p><b>Axes:</b> horizontal = how many <b>days ahead</b> we forecast (1-24); vertical = <b>forecast
   error</b>, lower is better. The dashed line at <b>1.0 = no skill</b> (no better than guessing the average).</p>
  <p>All lines rise (error grows the further ahead you look - expected). The key moment is where the
   grey dashed baseline <b>crosses 1.0 at about day 15</b> - persistence becomes useless. The teal
   line stays well below it all the way to day 24: it still has skill where the dumb baseline has none.</p>
  <p><b>The point:</b> teal and orange sit almost on top of each other, so the quantum model
   <em>ties</em> the classical one - we claim no quantum speed-up. The real win is over persistence,
   and it grows with lead.</p></div>
 <div class="card verdict"><strong>Verdict:</strong> the QRC beats persistence at 23 of 24 days
  ahead, with a widening margin. Honest note: vs strong classical models of equal size it's a
  <strong>tie</strong> - no "quantum advantage" is claimed (that would be false at this scale), and we say so.</div>
 <div class="read"><span class="h">Stress-test against a second classical method</span>
  <p>Beating one baseline can be luck, so we also ran <b>NVAR</b> (next-generation reservoir
   computing) - a different, cheap, seed-free classical method that usually beats echo-state networks.
   Result over 24 days ahead: <b>QRC mean error 0.633, NVAR 0.653, classical ESN 0.648, persistence
   0.792</b>. The quantum model holds <b>parity with both</b> strong classical methods (it edges them
   slightly on average but not by a meaningful margin) - so the honest claim is <b>parity, tested
   against two independent controls</b>, not one.</p></div>

 <h2>Step 2 - the detector beats the simple rules</h2>
 <p>Framed as genuine early warning - "will a heatwave <em>start</em> in the next 7 days?" - predicted
  from precursors, not from today's temperature (which would be circular). It scores
  <strong>2.6&times; better than chance</strong> and edges the best single-signal baseline. The
  precursors that matter, in order: <strong>30-day ocean warming trend, El Nino state, and rainfall</strong>
  - all physically sensible.</p>
 <p><small>Cross-checked on an independent labelled temperature benchmark (NAB): the same method
  recovers those anomalies too, so it isn't just re-learning our own labels.</small></p>

 <h2>Compose - the end-to-end warning</h2>
 <p>Running the detector on the forecast: a <strong>3-day warning catches 84% of real heatwave days</strong>,
  and the quantum-forecast warning beats the naive baseline in the 2-6 day window. Beyond a week the
  single "average" forecast smooths out and starts missing events - an honest limit we then fixed.</p>
 <figure><img src="{FIG_COMPOSE}" alt="Composed warning precision/recall and F1 vs lead">
  <figcaption>Left: precision &amp; recall of the composed warning vs lead. Right: overall F1. The
  quantum-forecast warning (solid) leads at short-mid range; the average forecast fades at long lead.</figcaption></figure>
 <div class="read"><span class="h">How to read this graph</span>
  <p><b>Left panel</b> - horizontal = days ahead, vertical = score (0-1). <b>Teal solid = recall</b>
   (of real heatwave days, what fraction did we catch) - starts near 0.97 at 1 day. <b>Grey = precision</b>
   (when we alarm, how often we're right). Dashed = the naive baseline for each.</p>
  <p><b>Right panel</b> - "F1" combines precision and recall into one score. <b>Solid red = our
   quantum-forecast warning; dashed red = the naive baseline.</b> The solid line is <b>above</b> the
   dashed one for days 2-6 (our forecast adds value), then they cross and it dips below - a single
   "average" forecast smooths out and starts missing events at long lead.</p>
  <p><b>The point:</b> a 3-day warning catches 84% of heatwave days; the quantum forecast wins the
   actionable 2-6 day window. The long-lead dip is the honest limit we fix in the next graph.</p></div>

 <h2>Step 3 - sampling many futures fixes the long-lead miss</h2>
 <p>The fix: instead of forecasting one <em>average</em> future, sample <strong>120 possible futures</strong>
  and raise the alarm if enough of them cross the heatwave line. Two things had to hold, and both do:</p>
 <ul>
  <li><strong>Honest probabilities</strong> - the ensemble's 90% band contains the truth 91% of the
   time (it's calibrated, not over-confident).</li>
  <li><strong>Recovered warning</strong> - at 14 days ahead the ensemble catches 40% of heatwave days
   vs the single forecast's 30%; the ensemble now wins overall.</li>
 </ul>
 <figure><img src="{FIG_STEP3}" alt="Ensemble calibration and recovered recall vs lead">
  <figcaption>Left: coverage tracks the nominal 90% line (calibrated). Right: the sampled-ensemble
  recall (solid) rises above the single-forecast recall (dashed) at long lead - warning recovered.</figcaption></figure>
 <div class="read"><span class="h">How to read this graph</span>
  <p><b>Left panel (calibration)</b> - the dotted line at <b>0.90</b> is the target; the teal line
   sits right on it. "Coverage" means: when the model says "90% confident the temperature is in this
   band", does the truth actually land inside 90% of the time? Yes - so the probabilities are
   <b>honest, not over-confident</b>. This gate had to pass before trusting any alarm.</p>
  <p><b>Right panel (recovered warning)</b> - <b>solid red = the sampled ensemble</b>; <b>dashed red =
   the single average forecast</b> (from the previous graph); dotted grey = persistence. Past day 8
   the solid line pulls <b>above</b> the dashed one - the ensemble catches events the single forecast
   missed (at 14 days: 40% vs 30%).</p>
  <p><b>The point:</b> sampling many futures does two things at once - keeps the probabilities honest
   (left) and restores the long-range warning (right).</p></div>

 <h2>Honest scorecard</h2>
 <table>
  <tr><th>Question</th><th>Result</th></tr>
  <tr><td>Forecast the temperature <em>anomaly</em>?</td><td class="win">Wins - skill to 24 days, beats persistence 23/24</td></tr>
  <tr><td>Beat a same-size <em>classical</em> model?</td><td class="tie">Tie - no quantum-advantage claim</td></tr>
  <tr><td>Detect heatwave onset from precursors?</td><td class="win">2.6&times; better than chance, beats baselines</td></tr>
  <tr><td>End-to-end warning, short-mid lead?</td><td class="win">3-day warning catches 84% of heatwave days</td></tr>
  <tr><td>End-to-end warning, long lead?</td><td class="win">Recovered by sampling futures (Step 3)</td></tr>
  <tr><td>Forecast raw temperature?</td><td class="hard">Hard - persistence is near-unbeatable there</td></tr>
 </table>
 <div class="card"><strong>Why it's credible:</strong> every result ships with the baseline that could
  beat it, and when the baseline wins we say so - the raw-temperature tie, the classical parity, the
  long-lead miss. A warning system nobody can poke a hole in is worth more than an inflated number.</div>
 <p><small>All figures and numbers are reproducible from the repository scripts
  (<code>label_anomalies.py</code>, <code>seed_sweep.py</code>, <code>detect.py</code>,
  <code>compose.py</code>, <code>stochastic.py</code>). 8-qubit exact simulation; no hardware claim.</small></p>
</div>
"""

out = HERE / "results_report.html"
out.write_text(HTML)
print(f"wrote {out} ({len(HTML)//1024} KB)")
