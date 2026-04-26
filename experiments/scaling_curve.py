"""Fit AvA and MHA accuracy as a function of parameters, then project.

We have multiple data points across scales (605K, 8M, 11M, 14M, 17M, 44M).
We fit:
    error(N) = E_inf + A * N^(-alpha)
to AvA and MHA separately for the multi-hop task. This gives us:
    - asymptotic error E_inf (the irreducible loss)
    - decay exponent alpha (how fast accuracy improves with scale)

If AvA's E_inf is significantly lower than MHA's, that's empirical
evidence the architectural ceiling differs at scale.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent.parent


def load_multihop_points():
    """Read multi-hop runs from all results files; return per-method points
    list of (params, mean_ood, std_ood)."""
    rows = []
    for fname in ("results.jsonl", "results_tuned.jsonl",
                  "results_multihop.jsonl", "results_multihop_deep.jsonl"):
        p = HERE / "results" / "gpu" / fname
        if p.exists():
            for ln in open(p):
                if not ln.strip(): continue
                r = json.loads(ln)
                if r.get("task") == "multihop" and "ood_acc" in r:
                    rows.append(r)
    print(f"Loaded {len(rows)} multi-hop runs from GPU results.")

    # Group by (attention, d_model, n_layers, n_train) — keep AvA at "primary"
    # configs, plus MHA same-d_model, plus the single MHA-wide that matches
    # AvA's params.
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        key = (r["attention"], r["d_model"], r["n_layers"], r["n_train"])
        grouped[key].append(r["ood_acc"])

    # Aggregate
    summary = []
    for k, accs in grouped.items():
        attn, d, L, n_train = k
        # Pick params from first row matching key.
        params = next(r["n_params"] for r in rows
                      if (r["attention"], r["d_model"], r["n_layers"], r["n_train"]) == k)
        summary.append({
            "attention": attn, "d": d, "L": L, "n_train": n_train,
            "params": params, "mean": float(np.mean(accs)),
            "std": float(np.std(accs)), "n": len(accs),
        })
    return summary


def power_law(N, E_inf, A, alpha):
    """Error as a function of params: E_inf + A * N^(-alpha)."""
    return E_inf + A * np.power(N, -alpha)


def fit_curve(points, label):
    if len(points) < 3:
        print(f"  too few points for {label}: skipping fit")
        return None
    xs = np.array([p["params"] for p in points], dtype=float)
    accs = np.array([p["mean"] for p in points], dtype=float)
    errs = 1.0 - accs
    try:
        popt, _ = curve_fit(power_law, xs, errs, p0=(0.1, 1.0, 0.3),
                             bounds=([0, 0, 0], [1, 1e6, 2]),
                             maxfev=5000)
        E_inf, A, alpha = popt
        residuals = errs - power_law(xs, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((errs - errs.mean())**2)
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        print(f"  {label}: E_inf={E_inf:.3f}  A={A:.3f}  alpha={alpha:.3f}  R^2={r2:.3f}")
        return dict(E_inf=E_inf, A=A, alpha=alpha, r2=r2, xs=xs.tolist(), accs=accs.tolist())
    except Exception as e:
        print(f"  fit failed for {label}: {e}")
        return None


def main():
    summary = load_multihop_points()
    # Print per-config.
    print("\nMulti-hop configurations:")
    for s in sorted(summary, key=lambda x: (x["attention"], x["params"])):
        print(f"  {s['attention']:3s} d={s['d']:>4} L={s['L']:>2} n_train={s['n_train']:>6}  "
              f"params={s['params']:>10,}  acc={s['mean']:.3f}±{s['std']:.3f}  n={s['n']}")

    # AvA at d=256 only -- the depth sweep is the cleanest scaling axis.
    # The d=512 L=8 recipe-A point fails for reasons unrelated to scale and
    # would distort the fit; we exclude it here (it's reported separately).
    ava_pts = [s for s in summary if s["attention"] == "ava" and s["d"] == 256]
    mha_pts = [s for s in summary if s["attention"] == "mha" and s["d"] == 256]
    # Add the wider MHA at d=356 too, since it's the param-matched comparator.
    mha_wide_pts = [s for s in summary if s["attention"] == "mha" and s["d"] == 356]
    ava_pts.sort(key=lambda x: x["params"])
    mha_pts.sort(key=lambda x: x["params"])
    mha_wide_pts.sort(key=lambda x: x["params"])

    print("\nFitting power laws (error = E_inf + A * N^-alpha):")
    ava_fit = fit_curve(ava_pts, "AvA d=256 sweep")
    mha_fit = fit_curve(mha_pts, "MHA d=256 sweep")
    mha_wide_fit = fit_curve(mha_wide_pts, "MHA wide d=356 sweep")

    # Plot.
    fig, ax = plt.subplots(figsize=(6, 4))
    for pts, color, marker, label in (
        (ava_pts, "#1f77b4", "o", "AvA (d=256)"),
        (mha_pts, "#d62728", "s", "MHA (d=256)"),
        (mha_wide_pts, "#ff9900", "^", "MHA-wide (d=356)"),
    ):
        if pts:
            xs = [p["params"] for p in pts]; ys = [p["mean"] for p in pts]
            es = [p["std"] for p in pts]
            ax.errorbar(xs, ys, yerr=es, marker=marker, color=color, label=label,
                         capsize=3, ls="None")
    # Projection.
    proj_xs = np.geomspace(2e6, 5e9, 80)
    for fit, color, label in (
        (ava_fit, "#1f77b4", "AvA fit"),
        (mha_fit, "#d62728", "MHA fit"),
        (mha_wide_fit, "#ff9900", "MHA-wide fit"),
    ):
        if fit:
            proj_err = power_law(proj_xs, fit["E_inf"], fit["A"], fit["alpha"])
            ax.plot(proj_xs, 1 - proj_err, color=color, ls="--", alpha=0.6,
                     label=f"{label} (E$_\\infty$={fit['E_inf']:.2f})")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Multi-hop OOD accuracy")
    ax.set_title("Power-law scaling on 2-hop binding")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0.3, 1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = HERE / "paper" / "fig_scaling_curve.pdf"
    fig.savefig(out)
    print(f"Wrote {out}")
    out_png = HERE / "paper" / "fig_scaling_curve.png"
    fig.savefig(out_png, dpi=150)

    # Save fits.
    save = HERE / "results" / "scaling_fits.json"
    with open(save, "w") as f:
        json.dump({"ava": ava_fit, "mha": mha_fit, "mha_wide": mha_wide_fit,
                   "summary": summary}, f, indent=2)
    print(f"Wrote {save}")


if __name__ == "__main__":
    main()
