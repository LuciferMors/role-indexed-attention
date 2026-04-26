"""Aggregate battery results and fill the \\textsc{[...]} placeholders in main.tex."""

from __future__ import annotations

import json
import re
import statistics as stat
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def load_battery() -> list[dict]:
    rows = []
    for fname in ("battery.jsonl", "strict_battery.jsonl"):
        p = HERE / "results" / fname
        if not p.exists():
            continue
        with open(p) as f:
            rows += [json.loads(l) for l in f]
    return rows


def agg(rows: list[dict]) -> dict[str, tuple[float, float]]:
    """Return {metric: (mean, std)} for iid, ood, params."""
    iid = [r["iid"] for r in rows]
    ood = [r["ood"] for r in rows]
    params = rows[0]["params"]
    return {
        "iid": (stat.mean(iid), stat.stdev(iid) if len(iid) > 1 else 0.0),
        "ood": (stat.mean(ood), stat.stdev(ood) if len(ood) > 1 else 0.0),
        "params": (params, 0),
    }


def fmt_pct(mean: float, std: float) -> str:
    return f"{100*mean:.1f} $\\pm$ {100*std:.1f}"


def fmt_params(n: int) -> str:
    return f"{n:,}"


def main():
    rows = load_battery()
    # Group by (experiment, tag), collect across seeds.
    groups = defaultdict(list)
    for r in rows:
        groups[(r["experiment"], r["tag"], r.get("n_train"))].append(r)

    # Print every group.
    print("=== Aggregated battery results ===")
    for key, rs in sorted(groups.items()):
        a = agg(rs)
        print(f"{key} n_seeds={len(rs)}  params={fmt_params(a['params'][0])}  "
              f"iid={fmt_pct(*a['iid'])}  ood={fmt_pct(*a['ood'])}")

    # Build substitutions for main.tex.
    paper_path = HERE / "paper" / "main.tex"
    paper = paper_path.read_text()

    # Main table (same d_model = 128).
    main_order = ["mha_d128", "ava_K1_d128", "ava_K2_d128", "ava_K4_d128"]
    main_rows = []
    for tag in main_order:
        rs = groups.get(("main", tag, 150000), [])
        if not rs:
            print(f"WARN missing main/{tag}")
            continue
        a = agg(rs)
        main_rows.append((tag, a))

    matched_order = ["mha_d176_match", "ava_K4_d128"]
    matched_rows = []
    for tag in matched_order:
        rs = groups.get(("param_matched", tag, 150000), [])
        if not rs:
            print(f"WARN missing param_matched/{tag}")
            continue
        a = agg(rs)
        matched_rows.append((tag, a))

    # Sample-efficiency: OOD at each budget.
    budgets = [5000, 15000, 50000, 150000]
    eff_rows = {"mha_d128": {}, "ava_K4_d128": {}}
    for budget in budgets:
        for tag in eff_rows:
            rs = groups.get(("sample_eff", tag, budget), [])
            if rs:
                a = agg(rs)
                eff_rows[tag][budget] = a["ood"]
            else:
                eff_rows[tag][budget] = (float("nan"), float("nan"))

    # ---- Build replacement tables ----
    def build_main_table() -> str:
        names = {
            "mha_d128": "MHA ($H{=}4$)",
            "ava_K1_d128": "AvA ($R{=}4, K{=}1$)",
            "ava_K2_d128": "AvA ($R{=}4, K{=}2$)",
            "ava_K4_d128": "AvA ($R{=}4, K{=}4$) \\textbf{(ours)}",
        }
        lines = []
        for tag, a in main_rows:
            lines.append(f"{names[tag]}  & {fmt_params(a['params'][0])}  & {fmt_pct(*a['iid'])}  & {fmt_pct(*a['ood'])} \\\\")
        return "\n".join(lines)

    def build_matched_table() -> str:
        names = {
            "mha_d176_match": "MHA wide ($H{=}4, d{=}176$)",
            "ava_K4_d128": "AvA ($R{=}4, K{=}4, d{=}128$)",
        }
        lines = []
        for tag, a in matched_rows:
            lines.append(f"{names[tag]}  & {fmt_params(a['params'][0])}  & {fmt_pct(*a['iid'])}  & {fmt_pct(*a['ood'])} \\\\")
        return "\n".join(lines)

    def build_eff_table() -> str:
        names = {"mha_d128": "MHA", "ava_K4_d128": "AvA"}
        lines = []
        for tag in ["mha_d128", "ava_K4_d128"]:
            row = names[tag] + " & " + " & ".join(fmt_pct(*eff_rows[tag][b]) for b in budgets) + " \\\\"
            lines.append(row)
        return "\n".join(lines)

    main_block = build_main_table()
    matched_block = build_matched_table()
    eff_block = build_eff_table()

    # Patch main.tex: swap the three table bodies.
    def replace_between(src: str, begin_marker: str, end_marker: str, block: str) -> str:
        # We replace the body between \midrule and \bottomrule following begin_marker label.
        idx = src.find(begin_marker)
        assert idx != -1, f"couldn't find {begin_marker}"
        mid = src.find("\\midrule", idx)
        bot = src.find("\\bottomrule", mid)
        assert mid != -1 and bot != -1
        return src[: mid + len("\\midrule")] + "\n" + block + "\n" + src[bot:]

    paper = replace_between(paper, "\\label{tab:main}", "\\bottomrule", main_block)
    paper = replace_between(paper, "\\label{tab:matched}", "\\bottomrule", matched_block)
    paper = replace_between(paper, "\\label{tab:sample-eff}", "\\bottomrule", eff_block)

    # Strict-OOD table.
    strict_groups = {
        "mha_d128": [r for r in rows if r.get("experiment") == "strict_ood" and r["tag"] == "mha_d128"],
        "mha_d176_match": [r for r in rows if r.get("experiment") == "strict_ood" and r["tag"] == "mha_d176_match"],
        "ava_K4_d128": [r for r in rows if r.get("experiment") == "strict_ood" and r["tag"] == "ava_K4_d128"],
    }
    if all(strict_groups.values()):
        names = {
            "mha_d128": "MHA ($H{=}4, d{=}128$)",
            "mha_d176_match": "MHA wide ($H{=}4, d{=}176$)",
            "ava_K4_d128": "AvA ($R{=}4, K{=}4$) \\textbf{(ours)}",
        }
        order = ["mha_d128", "mha_d176_match", "ava_K4_d128"]
        strict_lines = []
        strict_substitutions = {}
        for tag in order:
            a = agg(strict_groups[tag])
            strict_lines.append(f"{names[tag]}  & {fmt_params(a['params'][0])}  & {fmt_pct(*a['iid'])}  & {fmt_pct(*a['ood'])} \\\\")
            # Substitution keys for abstract.
            short = {"mha_d128": "STRICT_MHA", "mha_d176_match": "STRICT_MHA176",
                     "ava_K4_d128": "STRICT_AVA"}[tag]
            strict_substitutions[f"{short}_IID"] = f"{100*a['iid'][0]:.1f} $\\pm$ {100*a['iid'][1]:.1f}"
            strict_substitutions[f"{short}_OOD"] = f"{100*a['ood'][0]:.1f} $\\pm$ {100*a['ood'][1]:.1f}"
        strict_block = "\n".join(strict_lines)
        paper = replace_between(paper, "\\label{tab:strict}", "\\bottomrule", strict_block)

        # Abstract substitutions: just the OOD percentages without ± (terser).
        # NOTE: LaTeX escapes underscores as \_ inside \textbf{...}.
        abs_keys = {
            r"STRICT\_AVA\_OOD": f"{100*agg(strict_groups['ava_K4_d128'])['ood'][0]:.1f}",
            r"STRICT\_MHA\_OOD": f"{100*agg(strict_groups['mha_d128'])['ood'][0]:.1f}",
        }
        for placeholder, val in abs_keys.items():
            paper = paper.replace(f"\\textbf{{[{placeholder}]\\%}}", f"\\textbf{{{val}\\%}}")

    # Fill abstract numbers.
    ava_main = next((a for tag, a in main_rows if tag == "ava_K4_d128"), None)
    mha_main = next((a for tag, a in main_rows if tag == "mha_d128"), None)
    if ava_main and mha_main:
        # NOTE: re.sub treats backslashes in the replacement specially. Escape them.
        ava_repl = f"\\textbf{{{100*ava_main['iid'][0]:.1f} / {100*ava_main['ood'][0]:.1f}\\%}}"
        mha_repl = f"\\textbf{{{100*mha_main['iid'][0]:.1f} / {100*mha_main['ood'][0]:.1f}\\%}}"
        paper = paper.replace("\\textbf{[100.0 / 100.0]\\%}", ava_repl)
        paper = paper.replace("\\textbf{[74.4 / 74.3]\\%}", mha_repl)

    paper_path.write_text(paper)
    print(f"\nWrote filled paper -> {paper_path}")


if __name__ == "__main__":
    main()
