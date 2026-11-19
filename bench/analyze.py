"""Compute every number, table, and figure in the rebuilt paper.

Reads results/rebuild/{static,sast,llm_*}_{heval,swe}.jsonl and writes
  results/rebuild/summary.json      all metrics with bootstrap CIs
  paper/tables/*.tex                tables included by paper/main.tex
  paper/figures/*.pdf               ROC/PR curves
Bootstrap: 2000 resamples of clusters (HumanEvalFix pair / SWE-bench task), seed 0.
"""
import json, math, itertools
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "rebuild"
TAB = ROOT / "paper" / "tables"
FIG = ROOT / "paper" / "figures"
AGENTS = ["correctness", "security", "performance", "style"]
ABBR = {"correctness": "C", "security": "S", "performance": "P", "style": "St"}
WEIGHTS = {"security": 0.45, "correctness": 0.35, "performance": 0.15, "style": 0.05}  # CodeX-Verify's weights
SEV_W = {"LOW": 0.1, "MEDIUM": 0.3, "HIGH": 0.7, "CRITICAL": 1.0}
MODELS = {"claude-sonnet-5-5": "Sonnet 5.5", "claude-haiku-5-5": "Haiku 5.5"}
B = 2000
FULL = "+".join(AGENTS)


def load(name):
    p = RES / name
    return [json.loads(l) for l in open(p)] if p.exists() else []


def load_llm(model, ds):
    out = defaultdict(dict)
    for r in load(f"llm_{model}_{ds}.jsonl"):
        if "error" in r or "p_defect" not in r:
            continue
        out[r["id"]][r["role"]] = r
    return out


# ---------------------------------------------------------------- systems

def systems_for(ds):
    """Return (items, systems) where items is a list of {id,label,cluster,generator}
    and systems maps name -> {"score": {id: float}, "flag": {id: bool}, "group": str}."""
    static = {r["id"]: r for r in load(f"static_{ds}.jsonl")}
    sast = {r["id"]: r for r in load(f"sast_{ds}.jsonl")}
    items = [{"id": i, "label": r["label"], "cluster": r["pair"], "generator": r.get("generator")}
             for i, r in static.items()]
    S = {}

    def add(name, group, score, flag):
        S[name] = {"group": group, "score": score, "flag": flag}

    ids = list(static)
    add("Always flag", "trivial", {i: 1.0 for i in ids}, {i: True for i in ids})
    # CodeX-Verify (system's own aggregator), both operating points
    sc = {i: 1 - static[i]["subsets"][FULL]["score"] for i in ids}
    add("CodeX-Verify (FAIL)", "static", sc, {i: static[i]["subsets"][FULL]["status"] == "FAIL" for i in ids})
    add("CodeX-Verify (FAIL or WARN)", "static", sc, {i: static[i]["subsets"][FULL]["status"] != "PASS" for i in ids})
    # majority vote of the four static agents, each agent's verdict from its solo aggregator
    votes = {i: sum(static[i]["subsets"][a]["status"] != "PASS" for a in AGENTS) for i in ids}
    add("Static agents, majority vote", "static", {i: float(v) for i, v in votes.items()},
        {i: v >= 3 for i, v in votes.items()})
    if ds == "swe":
        def dscore(r):
            return sum(SEV_W.get(s, 0.3) * n for a in AGENTS for t, s, n in r["diff_new_issues"][a])
        add("CodeX-Verify, patch-aware", "static", {i: dscore(static[i]) for i in ids},
            {i: any(s in ("HIGH", "CRITICAL") for a in AGENTS for t, s, n in static[i]["diff_new_issues"][a])
             for i in ids})
    for tool in ("semgrep", "bandit"):
        if sast:
            add(tool.capitalize(), "sast", {i: sast[i][tool]["score"] for i in ids},
                {i: sast[i][tool]["score"] > 0 for i in ids})
    for model, mname in MODELS.items():
        L = load_llm(model, ds)
        have = [i for i in ids if "single" in L.get(i, {})]
        if len(have) < len(ids):
            continue
        add(f"{mname} single judge", "llm",
            {i: L[i]["single"]["p_defect"] / 100 for i in ids},
            {i: L[i]["single"]["verdict"] == "REJECT" for i in ids})
        if all(all(a in L[i] for a in AGENTS) for i in ids):
            p = {i: {a: L[i][a]["p_defect"] / 100 for a in AGENTS} for i in ids}
            rj = {i: {a: L[i][a]["verdict"] == "REJECT" for a in AGENTS} for i in ids}
            add(f"{mname} correctness agent", "llm", {i: p[i]["correctness"] for i in ids},
                {i: rj[i]["correctness"] for i in ids})
            add(f"{mname} 4 agents, mean", "llm-ma", {i: np.mean(list(p[i].values())) for i in ids},
                {i: np.mean(list(p[i].values())) >= 0.5 for i in ids})
            add(f"{mname} 4 agents, max", "llm-ma", {i: max(p[i].values()) for i in ids},
                {i: max(p[i].values()) >= 0.5 for i in ids})
            add(f"{mname} 4 agents, weighted", "llm-ma",
                {i: sum(WEIGHTS[a] * p[i][a] for a in AGENTS) for i in ids},
                {i: sum(WEIGHTS[a] * p[i][a] for a in AGENTS) >= 0.5 for i in ids})
            add(f"{mname} 4 agents, majority vote", "llm-ma",
                {i: sum(rj[i].values()) + 0.01 * np.mean(list(p[i].values())) for i in ids},
                {i: sum(rj[i].values()) >= 3 for i in ids})
            add(f"{mname} 4 agents, any reject", "llm-ma",
                {i: sum(rj[i].values()) + 0.01 * np.mean(list(p[i].values())) for i in ids},
                {i: any(rj[i].values()) for i in ids})
    # compute-matched control: the single judge's score averaged over repeated runs (SWE only)
    if ds == "swe":
        reps = []
        for tag in ("", "_run2", "_run3"):
            p = RES / f"llm_claude-sonnet-5-5_swe{tag}.jsonl"
            if p.exists():
                d = {}
                for l in open(p):
                    r = json.loads(l)
                    if r.get("role") == "single" and "p_defect" in r:
                        d[r["id"]] = r["p_defect"] / 100
                if all(i in d for i in ids):
                    reps.append(d)
        if len(reps) == 3:
            m = {i: float(np.mean([d[i] for d in reps])) for i in ids}
            add("Sonnet 5.5 single judge, mean of 3 runs", "llm", m, {i: m[i] >= 0.5 for i in ids})
    return items, S, static


# ---------------------------------------------------------------- metrics

def point(y, f):
    y, f = np.asarray(y, bool), np.asarray(f, bool)
    tp, fp = (f & y).sum(), (f & ~y).sum()
    fn, tn = (~f & y).sum(), (~f & ~y).sum()
    tpr = tp / max(tp + fn, 1); fpr = fp / max(fp + tn, 1)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return {"acc": (tp + tn) / len(y), "tpr": tpr, "fpr": fpr, "bal_acc": (tpr + 1 - fpr) / 2,
            "precision": prec, "f1": f1, "mcc": mcc}


def safe_auc(y, s):
    return roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")


def boot_idx(clusters, rng):
    uc = sorted(set(clusters))
    by = defaultdict(list)
    for k, c in enumerate(clusters):
        by[c].append(k)
    for _ in range(B):
        pick = rng.choice(len(uc), len(uc))
        yield np.array([k for j in pick for k in by[uc[j]]])


def ci(vals):
    v = np.array([x for x in vals if not np.isnan(x)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def evaluate(items, S, subset_ids=None):
    its = [it for it in items if subset_ids is None or it["id"] in subset_ids]
    ids = [it["id"] for it in its]
    y = np.array([it["label"] for it in its])
    cl = [it["cluster"] for it in its]
    rng = np.random.default_rng(0)
    idxs = list(boot_idx(cl, rng))
    out = {}
    for name, s in S.items():
        sc = np.array([s["score"][i] for i in ids]); fl = np.array([s["flag"][i] for i in ids])
        m = point(y, fl)
        m["auc"] = safe_auc(y, sc); m["ap"] = average_precision_score(y, sc)
        m["auc_ci"] = ci([safe_auc(y[k], sc[k]) for k in idxs])
        m["bal_acc_ci"] = ci([point(y[k], fl[k])["bal_acc"] for k in idxs])
        m["group"] = s["group"]
        out[name] = m
    out["_n"] = {"n": len(y), "pos": int(y.sum()), "neg": int(len(y) - y.sum())}
    return out


def paired_auc_diff(items, S, a, b, subset_ids=None):
    its = [it for it in items if subset_ids is None or it["id"] in subset_ids]
    ids = [it["id"] for it in its]
    y = np.array([it["label"] for it in its]); cl = [it["cluster"] for it in its]
    sa = np.array([S[a]["score"][i] for i in ids]); sb = np.array([S[b]["score"][i] for i in ids])
    d0 = safe_auc(y, sa) - safe_auc(y, sb)
    ds = [safe_auc(y[k], sa[k]) - safe_auc(y[k], sb[k]) for k in boot_idx(cl, np.random.default_rng(1))]
    ds = np.array([x for x in ds if not np.isnan(x)])
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return {"diff": d0, "ci": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))], "p_boot": float(p)}


def mcnemar(items, S, a, b, subset_ids=None):
    from scipy.stats import binomtest
    its = [it for it in items if subset_ids is None or it["id"] in subset_ids]
    ca = [S[a]["flag"][it["id"]] == bool(it["label"]) for it in its]
    cb = [S[b]["flag"][it["id"]] == bool(it["label"]) for it in its]
    n01 = sum(x and not z for x, z in zip(ca, cb)); n10 = sum(z and not x for x, z in zip(ca, cb))
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_only_correct": n01, "b_only_correct": n10, "p": p}


# ---------------------------------------------------------------- ablation, correlations, information

def ablation(items, static, subset_ids=None):
    its = [it for it in items if subset_ids is None or it["id"] in subset_ids]
    ids = [it["id"] for it in its]; y = np.array([it["label"] for it in its]); cl = [it["cluster"] for it in its]
    idxs = list(boot_idx(cl, np.random.default_rng(2)))
    rows, v = {}, {(): 0.5}
    for k in range(1, 5):
        for sub in itertools.combinations(AGENTS, k):
            key = "+".join(sub)
            sc = np.array([1 - static[i]["subsets"][key]["score"] for i in ids])
            fl = np.array([static[i]["subsets"][key]["status"] != "PASS" for i in ids])
            m = point(y, fl); m["auc"] = safe_auc(y, sc)
            m["auc_ci"] = ci([safe_auc(y[j], sc[j]) for j in idxs])
            rows[key] = m; v[sub] = m["auc"]
    n = 4
    shap = {}
    for a in AGENTS:
        tot = 0.0
        others = [x for x in AGENTS if x != a]
        for k in range(0, 4):
            for sub in itertools.combinations(others, k):
                with_a = tuple(x for x in AGENTS if x in sub or x == a)
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                tot += w * (v[with_a] - v[tuple(x for x in AGENTS if x in sub)])
        shap[a] = tot
    return rows, shap


def plugin_mi(xs, y):
    """Plug-in mutual information (bits) between a discrete tuple variable and binary y."""
    n = len(y); jx = Counter(zip(xs, y)); px = Counter(xs); py = Counter(y)
    return sum(c / n * math.log2((c / n) / ((px[x] / n) * (py[yy] / n))) for (x, yy), c in jx.items())


def cond_mi(a, b, y):
    out = 0.0
    for c in (0, 1):
        idx = [k for k in range(len(y)) if y[k] == c]
        if not idx:
            continue
        out += len(idx) / len(y) * plugin_mi([a[k] for k in idx], [b[k] for k in idx])
    return out


def correlations(items, agent_scores, agent_flags, subset_ids=None):
    its = [it for it in items if subset_ids is None or it["id"] in subset_ids]
    y = [it["label"] for it in its]
    res = {}
    for a, b in itertools.combinations(AGENTS, 2):
        per = {}
        for c in (0, 1):
            xa = [agent_scores[it["id"]][a] for it in its if it["label"] == c]
            xb = [agent_scores[it["id"]][b] for it in its if it["label"] == c]
            r = spearmanr(xa, xb).statistic if np.std(xa) > 0 and np.std(xb) > 0 else float("nan")
            per[c] = float(r)
        fa = [agent_flags[it["id"]][a] for it in its]; fb = [agent_flags[it["id"]][b] for it in its]
        res[f"{ABBR[a]}-{ABBR[b]}"] = {"rho_correct": per[0], "rho_buggy": per[1],
                                       "cmi_bits": cond_mi(fa, fb, y)}
    mi = {}
    for k in range(1, 5):
        for sub in itertools.combinations(AGENTS, k):
            xs = [tuple(agent_flags[it["id"]][a] for a in sub) for it in its]
            mi["+".join(ABBR[a] for a in sub)] = plugin_mi(xs, y)
    return res, mi


# ---------------------------------------------------------------- output helpers

def fmt(x, pct=True, d=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{100 * x:.{d}f}" if pct else f"{x:.{d}f}"


def write(name, text):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / name).write_text(text)


def main():
    summary = {}
    ev = {}
    data = {}
    for ds in ("heval", "swe"):
        items, S, static = systems_for(ds)
        data[ds] = (items, S, static)
    # evaluation subsets
    h_items, h_S, h_static = data["heval"]
    s_items, s_S, s_static = data["swe"]
    agent_ids = {it["id"] for it in s_items if it["generator"] != "gold"}
    gold_ids = {it["id"] for it in s_items if it["generator"] == "gold"}
    ev["heval"] = evaluate(h_items, h_S)
    ev["swe"] = evaluate(s_items, s_S, agent_ids)
    for g in sorted({it["generator"] for it in s_items} - {"gold"}):
        ev[f"swe_{g}"] = evaluate(s_items, s_S, {it["id"] for it in s_items if it["generator"] == g})
    # flag rate on merged human fixes (gold patches, all correct)
    ev["swe_gold_flag_rate"] = {name: float(np.mean([s["flag"][i] for i in gold_ids])) for name, s in s_S.items()}
    summary["eval"] = ev

    # within-pair discrimination on HumanEvalFix: does the score rank buggy above fixed?
    pairs = defaultdict(dict)
    for it in h_items:
        pairs[it["cluster"]][it["label"]] = it["id"]
    within = {}
    for name, s in h_S.items():
        d = [s["score"][p[1]] - s["score"][p[0]] for p in pairs.values()]
        within[name] = {"buggy_higher": float(np.mean([x > 0 for x in d])),
                        "tie": float(np.mean([x == 0 for x in d])),
                        "fixed_higher": float(np.mean([x < 0 for x in d]))}
    summary["heval_within_pair"] = within

    # paired comparisons (single judge vs multi-agent, static vs always-flag)
    comps = {}
    for ds, items, S, sub in (("heval", h_items, h_S, None), ("swe", s_items, s_S, agent_ids)):
        c = {}
        for m in MODELS.values():
            a = f"{m} single judge"
            if a not in S:
                continue
            for bname in [f"{m} 4 agents, mean", f"{m} 4 agents, max", f"{m} 4 agents, weighted",
                          f"{m} correctness agent"]:
                if bname in S:
                    c[f"{a} vs {bname}"] = {"auc": paired_auc_diff(items, S, a, bname, sub),
                                            "mcnemar": mcnemar(items, S, a, bname, sub)}
            if f"{m} single judge, mean of 3 runs" in S:
                c[f"{m} single judge, mean of 3 runs vs {m} 4 agents, mean"] = {
                    "auc": paired_auc_diff(items, S, f"{m} single judge, mean of 3 runs", f"{m} 4 agents, mean", sub),
                    "mcnemar": mcnemar(items, S, f"{m} single judge, mean of 3 runs", f"{m} 4 agents, mean", sub)}
            c[f"{a} vs CodeX-Verify"] = {"auc": paired_auc_diff(items, S, a, "CodeX-Verify (FAIL or WARN)", sub),
                                         "mcnemar": mcnemar(items, S, a, "CodeX-Verify (FAIL or WARN)", sub)}
        comps[ds] = c
    summary["paired"] = comps

    # ablation + Shapley (static)
    summary["ablation"] = {}
    for ds, items, static, sub in (("heval", h_items, h_static, None), ("swe", s_items, s_static, agent_ids)):
        rows, shap = ablation(items, static, sub)
        summary["ablation"][ds] = {"rows": rows, "shapley_auc": shap}

    # correlations + information (static agents and LLM agents)
    summary["correlation"] = {}
    for ds, items, static, sub in (("heval", h_items, h_static, None), ("swe", s_items, s_static, agent_ids)):
        sc = {i: {a: 1 - static[i]["subsets"][a]["score"] for a in AGENTS} for i in static}
        fl = {i: {a: static[i]["subsets"][a]["status"] != "PASS" for a in AGENTS} for i in static}
        corr, mi = correlations(items, sc, fl, sub)
        summary["correlation"][f"static_{ds}"] = {"pairs": corr, "mi_bits": mi}
        for model, mname in MODELS.items():
            L = load_llm(model, ds)
            if all(all(a in L.get(i, {}) for a in AGENTS) for i in static):
                sc = {i: {a: L[i][a]["p_defect"] / 100 for a in AGENTS} for i in static}
                fl = {i: {a: L[i][a]["verdict"] == "REJECT" for a in AGENTS} for i in static}
                corr, mi = correlations(items, sc, fl, sub)
                summary["correlation"][f"{mname}_{ds}"] = {"pairs": corr, "mi_bits": mi}

    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    make_tables(summary, data, agent_ids)
    make_figures(data, agent_ids)
    print(json.dumps({k: {n: (round(v["auc"], 3), round(v["bal_acc"], 3), round(v["tpr"], 3), round(v["fpr"], 3))
                          for n, v in e.items() if not n.startswith("_")} for k, e in ev.items()
                      if k in ("heval", "swe")}, indent=1))


MAIN_ROWS = ["Always flag", "CodeX-Verify (FAIL)", "CodeX-Verify (FAIL or WARN)", "Static agents, majority vote",
             "CodeX-Verify, patch-aware", "Semgrep", "Bandit",
             "Haiku 5.5 single judge", "Haiku 5.5 4 agents, mean", "Haiku 5.5 4 agents, majority vote",
             "Sonnet 5.5 single judge", "Sonnet 5.5 single judge, mean of 3 runs", "Sonnet 5.5 correctness agent", "Sonnet 5.5 4 agents, mean",
             "Sonnet 5.5 4 agents, max", "Sonnet 5.5 4 agents, weighted", "Sonnet 5.5 4 agents, majority vote",
             "Sonnet 5.5 4 agents, any reject"]


def make_tables(summary, data, agent_ids):
    ev = summary["eval"]
    # main table: both datasets side by side
    lines = [r"\begin{tabular}{@{}l ccc ccc c@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{HumanEvalFix (164 buggy / 164 fixed)} & \multicolumn{3}{c}{SWE-bench Verified agent patches} & Gold \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Verifier & AUC [95\% CI] & TPR & FPR & AUC [95\% CI] & TPR & FPR & flagged \\", r"\midrule"]
    prev = None
    for name in MAIN_ROWS:
        h = ev["heval"].get(name); s = ev["swe"].get(name)
        if h is None and s is None:
            continue
        grp = (h or s)["group"]
        if prev and grp.split("-")[0] != prev.split("-")[0]:
            lines.append(r"\midrule")
        prev = grp

        def cell(m):
            if m is None:
                return "\\multicolumn{3}{c}{n/a}"
            return f"{fmt(m['auc'], False, 2)} [{fmt(m['auc_ci'][0], False, 2)}, {fmt(m['auc_ci'][1], False, 2)}] & {fmt(m['tpr'])} & {fmt(m['fpr'])}"
        g = ev["swe_gold_flag_rate"].get(name)
        lines.append(f"{name} & {cell(h)} & {cell(s)} & {fmt(g) if g is not None else '--'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("main.tex", "\n".join(lines) + "\n")

    # ablation table (static subsets) both datasets
    ab = summary["ablation"]
    lines = [r"\begin{tabular}{@{}l cc cc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{HumanEvalFix} & \multicolumn{2}{c}{SWE-bench agent patches} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"Agents & AUC & TPR$-$FPR & AUC & TPR$-$FPR \\", r"\midrule"]
    for k in range(1, 5):
        for sub in itertools.combinations(AGENTS, k):
            key = "+".join(sub); lab = "+".join(ABBR[a] for a in sub)
            h = ab["heval"]["rows"][key]; s = ab["swe"]["rows"][key]
            lines.append(f"{lab} & {fmt(h['auc'], False, 2)} & {fmt(h['tpr'] - h['fpr'])} & {fmt(s['auc'], False, 2)} & {fmt(s['tpr'] - s['fpr'])} \\\\")
        if k < 4:
            lines.append(r"\midrule")
    lines += [r"\midrule", "Shapley value (AUC) & " + " & ".join([]) ]
    lines.pop()
    for a in AGENTS:
        lines.append(f"Shapley, {ABBR[a]} & {ab['heval']['shapley_auc'][a]:+.3f} & & {ab['swe']['shapley_auc'][a]:+.3f} & \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("ablation.tex", "\n".join(lines) + "\n")

    # correlation table
    C = summary["correlation"]
    keys = [k for k in ["static_heval", "static_swe", "Sonnet 5.5_heval", "Sonnet 5.5_swe"] if k in C]
    lab = {"static_heval": "Static, HEF", "static_swe": "Static, SWE", "Sonnet 5.5_heval": "LLM, HEF",
           "Sonnet 5.5_swe": "LLM, SWE"}
    lines = [r"\begin{tabular}{@{}l " + "c" * len(keys) + "@{}}", r"\toprule",
             "Agent pair & " + " & ".join(lab[k] for k in keys) + r" \\", r"\midrule"]
    for pair in C[keys[0]]["pairs"]:
        cells = []
        for k in keys:
            p = C[k]["pairs"][pair]
            r0, r1 = p["rho_correct"], p["rho_buggy"]
            cells.append(f"{fmt(r0, False, 2)} / {fmt(r1, False, 2)}")
        lines.append(f"{pair} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule"]
    for sub in ["C", "C+S", "C+S+P", "C+S+P+St"]:
        lines.append(f"$\\hat I(D_{{{sub}}};B)$ (bits) & " + " & ".join(f"{C[k]['mi_bits'][sub]:.3f}" for k in keys) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("correlation.tex", "\n".join(lines) + "\n")

    # per-generator table for the LLM judges (self-preference check)
    gens = [k for k in ev if k.startswith("swe_2")]
    lines = [r"\begin{tabular}{@{}l " + "cc" * len(gens) + "@{}}", r"\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{'Claude 3.5 Sonnet agent' if 'claude' in g else 'GPT-4o Agentless'} ({ev[g]['_n']['neg']} res./{ev[g]['_n']['pos']} unres.)}}" for g in gens) + r" \\",
             "Verifier & " + " & ".join("AUC & Bal.\\ acc." for _ in gens) + r" \\", r"\midrule"]
    for name in ["CodeX-Verify (FAIL or WARN)", "CodeX-Verify, patch-aware", "Haiku 5.5 single judge",
                 "Sonnet 5.5 single judge", "Sonnet 5.5 4 agents, mean"]:
        if name in ev[gens[0]]:
            lines.append(name + " & " + " & ".join(f"{fmt(ev[g][name]['auc'], False, 2)} & {fmt(ev[g][name]['bal_acc'])}" for g in gens) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("generators.tex", "\n".join(lines) + "\n")


def make_figures(data, agent_ids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    show = [("CodeX-Verify (FAIL or WARN)", "CodeX-Verify (4 static agents)", "#d95f02", "-"),
            ("CodeX-Verify, patch-aware", "CodeX-Verify, patch-aware", "#d95f02", ":"),
            ("Semgrep", "Semgrep", "#7570b3", "-"),
            ("Haiku 5.5 single judge", "Haiku 5.5, single", "#66a61e", "--"),
            ("Sonnet 5.5 single judge", "Sonnet 5.5, single", "#1b9e77", "-"),
            ("Sonnet 5.5 4 agents, mean", "Sonnet 5.5, 4 agents (mean)", "#1b9e77", "--")]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.1))
    for ax, ds, title in ((axes[0], "heval", "HumanEvalFix"), (axes[1], "swe", "SWE-bench Verified agent patches")):
        items, S, _ = data[ds]
        its = [it for it in items if ds == "heval" or it["id"] in agent_ids]
        y = [it["label"] for it in its]
        for name, lab, col, ls in show:
            if name not in S:
                continue
            sc = [S[name]["score"][it["id"]] for it in its]
            fpr, tpr, _ = roc_curve(y, sc)
            ax.plot(fpr, tpr, color=col, ls=ls, lw=1.4, label=f"{lab} ({roc_auc_score(y, sc):.2f})")
        ax.plot([0, 1], [0, 1], color="#999999", lw=0.8)
        ax.set_title(title, fontsize=9); ax.set_xlabel("False-positive rate", fontsize=8)
        ax.set_ylabel("True-positive rate", fontsize=8); ax.tick_params(labelsize=7)
        ax.legend(fontsize=5.5, loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "roc.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.1))
    for ax, ds, title in ((axes[0], "heval", "HumanEvalFix"), (axes[1], "swe", "SWE-bench Verified agent patches")):
        items, S, _ = data[ds]
        its = [it for it in items if ds == "heval" or it["id"] in agent_ids]
        y = [it["label"] for it in its]
        for name, lab, col, ls in show:
            if name not in S:
                continue
            sc = [S[name]["score"][it["id"]] for it in its]
            p, r, _ = precision_recall_curve(y, sc)
            ax.plot(r, p, color=col, ls=ls, lw=1.4, label=f"{lab} ({average_precision_score(y, sc):.2f})")
        ax.axhline(np.mean(y), color="#999999", lw=0.8)
        ax.set_title(title, fontsize=9); ax.set_xlabel("Recall", fontsize=8); ax.set_ylabel("Precision", fontsize=8)
        ax.tick_params(labelsize=7); ax.legend(fontsize=5.5, loc="lower left", frameon=False); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(FIG / "pr.pdf"); plt.close(fig)


if __name__ == "__main__":
    import sys as _sys
    if "--macros-only" not in _sys.argv:
        main()


# ---------------------------------------------------------------- in-text numbers as LaTeX macros

def macro_name(s):
    return "".join(ch for ch in s.title() if ch.isalpha())


def make_macros():
    """Every number quoted in the paper's prose is a macro defined here."""
    d = json.loads((RES / "summary.json").read_text())
    ev = d["eval"]; M = {}

    def put(k, v):
        M[k] = v

    def sysm(prefix, ds, name):
        m = ev[ds][name]
        put(prefix + "Auc", f"{m['auc']:.2f}")
        put(prefix + "AucLo", f"{m['auc_ci'][0]:.2f}"); put(prefix + "AucHi", f"{m['auc_ci'][1]:.2f}")
        put(prefix + "Tpr", f"{100 * m['tpr']:.1f}"); put(prefix + "Fpr", f"{100 * m['fpr']:.1f}")
        put(prefix + "Bal", f"{100 * m['bal_acc']:.1f}"); put(prefix + "Ap", f"{m['ap']:.2f}")

    for ds, tag in (("heval", "Hef"), ("swe", "Swe")):
        sysm(tag + "Cv", ds, "CodeX-Verify (FAIL or WARN)")
        sysm(tag + "CvFail", ds, "CodeX-Verify (FAIL)")
        sysm(tag + "Vote", ds, "Static agents, majority vote")
        sysm(tag + "Semgrep", ds, "Semgrep"); sysm(tag + "Bandit", ds, "Bandit")
        if "Sonnet 5.5 single judge, mean of 3 runs" in ev[ds]:
            sysm(tag + "SonnetThree", ds, "Sonnet 5.5 single judge, mean of 3 runs")
        for m in ("Sonnet", "Haiku"):
            sysm(tag + m, ds, f"{m} 5.5 single judge")
            sysm(tag + m + "Corr", ds, f"{m} 5.5 correctness agent")
            for agg in ("mean", "max", "weighted", "majority vote", "any reject"):
                sysm(tag + m + macro_name(agg), ds, f"{m} 5.5 4 agents, {agg}")
    sysm("SweCvDiff", "swe", "CodeX-Verify, patch-aware")
    g = ev["swe_gold_flag_rate"]
    put("GoldCv", f"{100 * g['CodeX-Verify (FAIL or WARN)']:.1f}"); put("GoldCvFail", f"{100 * g['CodeX-Verify (FAIL)']:.1f}")
    put("GoldSonnet", f"{100 * g['Sonnet 5.5 single judge']:.1f}"); put("GoldHaiku", f"{100 * g['Haiku 5.5 single judge']:.1f}")
    put("GoldCvDiff", f"{100 * g['CodeX-Verify, patch-aware']:.1f}")
    put("GoldSemgrep", f"{100 * g['Semgrep']:.1f}")
    w = d["heval_within_pair"]
    put("HefCvTie", f"{100 * w['CodeX-Verify (FAIL or WARN)']['tie']:.1f}")
    put("HefSonnetPairWin", f"{100 * w['Sonnet 5.5 single judge']['buggy_higher']:.1f}")
    for ds, tag in (("heval", "Hef"), ("swe", "Swe")):
        for k, v in d["paired"][ds].items():
            key = tag + "Diff" + macro_name(k.replace("5.5", "").replace("vs", "Vs"))
            put(key, f"{v['auc']['diff']:+.3f}"); put(key + "Abs", f"{abs(v['auc']['diff']):.3f}"); put(key + "Lo", f"{v['auc']['ci'][0]:+.3f}")
            put(key + "Hi", f"{v['auc']['ci'][1]:+.3f}"); put(key + "P", f"{v['auc']['p_boot']:.3f}")
        sh = d["ablation"][ds]["shapley_auc"]
        for a in AGENTS:
            put(tag + "Shap" + macro_name(a), f"{sh[a]:+.3f}")
        put(tag + "AblCP", f"{d['ablation'][ds]['rows']['correctness+performance']['auc']:.2f}")
        best = max(d["ablation"][ds]["rows"].items(), key=lambda kv: kv[1]["auc"])
        put(tag + "AblBest", f"{best[1]['auc']:.2f}"); put(tag + "AblWorst", f"{min(r['auc'] for r in d['ablation'][ds]['rows'].values()):.2f}")
    C = d["correlation"]
    for key, tag in (("static_heval", "StaticHef"), ("static_swe", "StaticSwe"), ("Sonnet 5.5_heval", "SonnetHef"),
                     ("Sonnet 5.5_swe", "SonnetSwe"), ("Haiku 5.5_swe", "HaikuSwe")):
        rhos = [x for p in C[key]["pairs"].values() for x in (p["rho_correct"], p["rho_buggy"]) if not np.isnan(x)]
        put(tag + "RhoMin", f"{min(rhos):.2f}"); put(tag + "RhoMax", f"{max(rhos):.2f}")
        put(tag + "MiC", f"{C[key]['mi_bits']['C']:.3f}"); put(tag + "MiAll", f"{C[key]['mi_bits']['C+S+P+St']:.3f}")
        put(tag + "MiBestSingle", f"{max(C[key]['mi_bits'][a] for a in ['C', 'S', 'P', 'St']):.3f}")
    # static-system internals on SWE snippets
    st = load("static_swe.jsonl")
    put("SweNItems", str(len(st)))
    put("SweSyntaxCrit", str(sum(any(t == "ast_analysis" and s == "CRITICAL" for t, s in r["agents"]["correctness"]["issues"]) for r in st)))
    put("SweExecHigh", str(sum(any(t == "execution" and s == "HIGH" for t, s in r["agents"]["correctness"]["issues"]) for r in st)))
    hs = load("static_heval.jsonl")
    put("HefPotentialBugBuggy", f"{100 * np.mean([any(t == 'potential_bug' for t, s in r['agents']['correctness']['issues']) for r in hs if r['label'] == 1]):.1f}")
    put("HefPotentialBugFixed", f"{100 * np.mean([any(t == 'potential_bug' for t, s in r['agents']['correctness']['issues']) for r in hs if r['label'] == 0]):.1f}")
    # repeat runs of the Sonnet single judge on SWE
    runs = []
    agent = {it for it in [json.loads(l)["id"] for l in open(ROOT / "bench/data/swe_labels.jsonl")]}
    lab = {json.loads(l)["id"]: (json.loads(l)["label"], json.loads(l)["generator"]) for l in open(ROOT / "bench/data/swe_labels.jsonl")}
    for tag in ("", "_run2", "_run3"):
        p = RES / f"llm_claude-sonnet-5-5_swe{tag}.jsonl"
        if not p.exists():
            continue
        sc = {}
        for l in open(p):
            r = json.loads(l)
            if r.get("role") == "single" and "p_defect" in r:
                sc[r["id"]] = r["p_defect"] / 100
        ids = [i for i in sc if lab[i][1] != "gold"]
        if len(ids) == 300:
            runs.append(roc_auc_score([lab[i][0] for i in ids], [sc[i] for i in ids]))
    put("SonnetRuns", str(len(runs)))
    put("SonnetRunMean", f"{np.mean(runs):.3f}"); put("SonnetRunSd", f"{np.std(runs, ddof=1) if len(runs) > 1 else 0:.3f}")
    # v1 re-analysis
    o = json.loads((RES / "old_set_leak.json").read_text())
    put("OldN", str(o["n"])); put("OldBuggy", str(o["buggy"])); put("OldMajority", f"{100 * o['majority_baseline_acc']:.1f}")
    put("OldBugComments", str(o["samples_with_#BUG"])); put("OldBugCommentsBuggy", str(o["buggy_with_#BUG"]))
    put("OldCvAcc", f"{100 * o['codexverify_FAILonly_code']['acc']:.1f}"); put("OldCvTpr", f"{100 * o['codexverify_FAILonly_code']['tpr']:.1f}")
    put("OldCvFpr", f"{100 * o['codexverify_FAILonly_code']['fpr']:.1f}")
    put("OldCvStripAcc", f"{100 * o['codexverify_FAILonly_stripped']['acc']:.1f}")
    put("OldSonnetAcc", f"{100 * o['sonnet_single_code']['acc']:.1f}"); put("OldSonnetStripAcc", f"{100 * o['sonnet_single_stripped']['acc']:.1f}")
    put("OldSonnetTpr", f"{100 * o['sonnet_single_code']['tpr']:.1f}"); put("OldSonnetStripTpr", f"{100 * o['sonnet_single_stripped']['tpr']:.1f}")
    for key, tag in (("Sonnet 5.5_swe", "SonnetSwe"), ("Haiku 5.5_swe", "HaikuSwe"), ("Sonnet 5.5_heval", "SonnetHef")):
        mi = C[key]["mi_bits"]
        put(tag + "MiGain", f"{mi['C+S+P+St'] - mi['C']:.3f}")
        for a in ("S", "P", "St"):
            put(tag + "Mi" + a, f"{mi[a]:.3f}")
    for g, tag in (("swe_20241022_tools_claude-3-5-sonnet-updated", "GenClaude"), ("swe_20241028_agentless-1.5_gpt4o", "GenGpt")):
        for name, t2 in (("Sonnet 5.5 single judge", "Sonnet"), ("Haiku 5.5 single judge", "Haiku")):
            put(tag + t2 + "Auc", f"{ev[g][name]['auc']:.2f}")
    full = ROOT / "bench/data/swe_patches.jsonl"  # rebuilt by bench/build_swe.py
    if full.exists():
        put("NTruncated", str(sum(len(json.loads(l)["patch"]) > 30000 for l in open(full))))
    lines = ["% generated by bench/analyze.py; do not edit"]
    for k, v in sorted(M.items()):
        lines.append(f"\\newcommand{{\\n{k}}}{{{v}}}")
    write("numbers.tex", "\n".join(lines) + "\n")
    return M


if __name__ == "__main__" and True:
    make_macros()
