"""Robustness analyses requested in review (no API calls).

1. Recall of the reference fix: how often LLM verifier reasons mention an upstream /
   reference / known fix, and judge AUC on items where they do not.
2. Trivial patch-feature baselines (length, changed lines, files touched, generator).
3. Patch-aware static AUC within each generator.
4. Cross-validated logistic stack of the four LLM agents' scores (grouped by task).
5. Permutation null for the information the S, P, St agents add beyond C:
   shuffle their verdicts within each class (keeping each agent's class-conditional
   rates, destroying dependence on C), recompute the plug-in gain. Miller-Madow
   corrected estimates are reported alongside the plug-in ones.
6. Holm-adjusted p-values for the single-judge vs four-agent comparisons.
7. Static-agent dependence on HumanEvalFix and on patch-aware SWE scores.
8. The earlier evaluation's SWE-bench Lite gold-patch run: share of snippets with a
   CRITICAL syntax issue.
9. API cost of all logged LLM calls at list prices.
Output: results/rebuild/robustness.json
"""
import json, math, re, itertools
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "rebuild"
AGENTS = ["correctness", "security", "performance", "style"]
ABBR = {"correctness": "C", "security": "S", "performance": "P", "style": "St"}
RECALL = re.compile(r"\b(upstream|reference (fix|patch|implementation|solution)|official (fix|patch)|actual fix|merged fix|known fix|the real fix)\b", re.I)
SEV_W = {"LOW": 0.1, "MEDIUM": 0.3, "HIGH": 0.7, "CRITICAL": 1.0}
PRICE = {"claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


def llm(model, ds, tag=""):
    out = defaultdict(dict)
    for l in open(RES / f"llm_{model}_{ds}{tag}.jsonl"):
        r = json.loads(l)
        if "p_defect" in r:
            out[r["id"]][r["role"]] = r
    return out


def plugin_mi(xs, y):
    n = len(y); jx = Counter(zip(xs, y)); px = Counter(xs); py = Counter(y)
    return sum(c / n * math.log2((c / n) / ((px[x] / n) * (py[yy] / n))) for (x, yy), c in jx.items())


def mm_mi(xs, y):
    """Miller-Madow bias-corrected MI = H_mm(X) + H_mm(Y) - H_mm(X,Y)."""
    n = len(y)

    def h(cnt):
        return -sum(c / n * math.log2(c / n) for c in cnt.values()) + (len(cnt) - 1) / (2 * n * math.log(2))
    return h(Counter(xs)) + h(Counter(y)) - h(Counter(zip(xs, y)))


def main():
    R = {}
    lab = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "bench/data/swe_labels.jsonl")}
    agent_ids = [i for i in lab if lab[i]["generator"] != "gold"]
    gold_ids = [i for i in lab if lab[i]["generator"] == "gold"]
    y = np.array([lab[i]["label"] for i in agent_ids])

    # 1. recall mentions
    rec = {}
    for model in PRICE:
        L = llm(model, "swe")
        m = {}
        for role in ["single"] + AGENTS:
            m[role] = {"agent": float(np.mean([bool(RECALL.search(L[i][role]["reason"])) for i in agent_ids])),
                       "gold": float(np.mean([bool(RECALL.search(L[i][role]["reason"])) for i in gold_ids]))}
        clean = [k for k, i in enumerate(agent_ids) if not RECALL.search(L[i]["single"]["reason"])]
        clean_all = [k for k, i in enumerate(agent_ids) if not any(RECALL.search(L[i][r]["reason"]) for r in ["single"] + AGENTS)]
        s1 = np.array([L[i]["single"]["p_defect"] for i in agent_ids])
        smean = np.array([np.mean([L[i][a]["p_defect"] for a in AGENTS]) for i in agent_ids])
        def auc_on(idx, s):
            idx = np.array(idx)
            return float(roc_auc_score(y[idx], s[idx])) if len(set(y[idx])) == 2 else float("nan")
        mentioned = [k for k in range(len(agent_ids)) if k not in set(clean)]
        rec[model] = {"rates": m, "n_clean_single": len(clean), "n_clean_all": len(clean_all),
                      "auc_single_all": auc_on(range(len(agent_ids)), s1),
                      "auc_single_clean": auc_on(clean, s1), "auc_single_mentioned": auc_on(mentioned, s1),
                      "auc_mean4_clean_all": auc_on(clean_all, smean), "auc_single_clean_all": auc_on(clean_all, s1),
                      "clean_all_pos": int(y[np.array(clean_all)].sum()) if clean_all else 0}
    R["recall"] = rec
    # heval mentions of HumanEval / canonical
    L = llm("claude-sonnet-5-5", "heval")
    R["recall_heval_sonnet"] = float(np.mean([bool(re.search(r"humaneval|canonical", L[i]["single"]["reason"], re.I)) for i in L]))

    # 2. trivial features (needs the full patch file for patch text)
    full = ROOT / "bench/data/swe_patches.jsonl"
    feats = {}
    if full.exists():
        P = {json.loads(l)["id"]: json.loads(l) for l in open(full)}
        def changed(p):
            return sum(1 for l in p.splitlines() if (l.startswith("+") or l.startswith("-")) and not l.startswith(("+++", "---")))
        F = {"patch length": [len(P[i]["patch"]) for i in agent_ids],
             "changed lines": [changed(P[i]["patch"]) for i in agent_ids],
             "files touched": [P[i]["patch"].count("diff --git") for i in agent_ids],
             "generator (GPT-4o agent)": [int("gpt4o" in lab[i]["generator"]) for i in agent_ids]}
        feats = {k: float(roc_auc_score(y, v)) for k, v in F.items()}
    R["trivial_features_auc"] = feats

    # 3. patch-aware static within generator
    st = {json.loads(l)["id"]: json.loads(l) for l in open(RES / "static_swe.jsonl")}
    def dscore(r):
        return sum(SEV_W.get(s, 0.3) * n for a in AGENTS for t, s, n in r["diff_new_issues"][a])
    pa = {}
    for g in sorted({lab[i]["generator"] for i in agent_ids}):
        ids = [i for i in agent_ids if lab[i]["generator"] == g]
        pa[g] = {"auc": float(roc_auc_score([lab[i]["label"] for i in ids], [dscore(st[i]) for i in ids])),
                 "mean_score": float(np.mean([dscore(st[i]) for i in ids]))}
    R["patch_aware_by_generator"] = pa

    # 4. logistic stack, grouped CV by task
    stack = {}
    groups = [lab[i]["pair"] for i in agent_ids]
    for model in PRICE:
        L = llm(model, "swe")
        X = np.array([[L[i][a]["p_defect"] / 100 for a in AGENTS] for i in agent_ids])
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            clf = LogisticRegression(C=1.0).fit(X[tr], y[tr])
            pred[te] = clf.predict_proba(X[te])[:, 1]
        stack[model] = float(roc_auc_score(y, pred))
    R["stack_cv_auc"] = stack

    # 5. permutation null for the gain of S,P,St over C (verdicts), agent patches
    rng = np.random.default_rng(0)
    perm = {}
    for model in PRICE:
        L = llm(model, "swe")
        V = {a: np.array([L[i][a]["verdict"] == "REJECT" for i in agent_ids]) for a in AGENTS}
        yl = list(y)
        def gain(V):
            xs_all = list(zip(*[V[a] for a in AGENTS])); xs_c = list(V["correctness"])
            return plugin_mi(xs_all, yl) - plugin_mi(xs_c, yl), mm_mi(xs_all, yl) - mm_mi(xs_c, yl)
        obs, obs_mm = gain(V)
        null = []
        for _ in range(1000):
            W = dict(V)
            for a in ["security", "performance", "style"]:
                w = V[a].copy()
                for c in (0, 1):
                    idx = np.where(y == c)[0]
                    w[idx] = w[rng.permutation(idx)]
                W[a] = w
            null.append(gain(W)[0])
        null = np.array(null)
        perm[model] = {"observed_gain": obs, "observed_gain_mm": obs_mm, "null_mean": float(null.mean()),
                       "null_p05": float(np.percentile(null, 5)), "p_le": float((null <= obs).mean())}
    R["permutation_gain"] = perm

    # 6. Holm on the single vs four-agent paired AUC p-values
    summ = json.loads((RES / "summary.json").read_text())
    tests = [(ds, k, v["auc"]["p_boot"]) for ds in ("heval", "swe") for k, v in summ["paired"][ds].items()
             if "4 agents" in k or "correctness agent" in k]
    order = sorted(range(len(tests)), key=lambda j: tests[j][2])
    m = len(tests); adj = [0.0] * m; run = 0.0
    for rank, j in enumerate(order):
        run = max(run, min(1.0, (m - rank) * tests[j][2])); adj[j] = run
    R["holm"] = [{"dataset": t[0], "comparison": t[1], "p": t[2], "p_holm": a} for t, a in zip(tests, adj)]

    # 7. static dependence on HEF and on patch-aware SWE scores
    sd = {}
    hs = [json.loads(l) for l in open(RES / "static_heval.jsonl")]
    for name, rows, score in (("heval", hs, lambda r, a: 1 - r["subsets"][a]["score"]),
                              ("swe_patch_aware", [st[i] for i in agent_ids],
                               lambda r, a: sum(SEV_W.get(s, 0.3) * n for t, s, n in r["diff_new_issues"][a]))):
        pairs = {}
        for a, b in itertools.combinations(AGENTS, 2):
            per = []
            for c in (0, 1):
                xa = [score(r, a) for r in rows if r["label"] == c]; xb = [score(r, b) for r in rows if r["label"] == c]
                per.append(float(spearmanr(xa, xb).statistic) if np.std(xa) > 0 and np.std(xb) > 0 else float("nan"))
            pairs[f"{ABBR[a]}-{ABBR[b]}"] = per
        yy = [r["label"] for r in rows]
        sd[name] = {"pairs": pairs,
                    "agent_auc": {a: float(roc_auc_score(yy, [score(r, a) for r in rows])) for a in AGENTS},
                    "constant_agents": [a for a in AGENTS if len({score(r, a) for r in rows}) == 1]}
    R["static_dependence"] = sd

    # 8. earlier Lite gold-patch run
    v1 = json.load(open(ROOT / "results/swe_bench_real_results_20251012_182345.json"))
    rows = v1["detailed_results"]
    R["v1_lite_gold"] = {"n": len(rows), "fail": sum(r["our_verdict"] == "FAIL" for r in rows),
                         "syntax_critical": sum(any(i.get("type") == "ast_analysis" and str(i.get("severity")).lower() == "critical"
                                                    for i in (r.get("detailed_issues") or [])) for r in rows)}

    # 9. cost
    cost = 0.0
    for f in RES.glob("llm_*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            p = PRICE[r["model"]]
            cost += (r.get("in_tok", 0) * p[0] + r.get("out_tok", 0) * p[1]) / 1e6
    R["logged_cost_usd"] = cost
    (RES / "robustness.json").write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
