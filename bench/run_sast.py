"""Semgrep and Bandit on the same benchmark inputs.

heval: each function is one file; score = severity-weighted finding count.
swe:   pre- and post-patch versions of every touched .py file; score = the
       severity-weighted count of findings present after the patch but not
       before (matched by rule id per file).
Output: results/rebuild/sast_<dataset>.jsonl
"""
import json, subprocess, sys, tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEMGREP_PACKS = ["p/python", "p/security-audit", "p/default", "p/r2c-best-practices", "p/r2c-bug-scan"]
W = {"INFO": 0.1, "LOW": 0.1, "WARNING": 0.3, "MEDIUM": 0.3, "ERROR": 0.7, "HIGH": 0.7, "CRITICAL": 1.0}


def semgrep(tree):
    cmd = ["semgrep", "--json", "--metrics=off", "--quiet", "--timeout", "30", "-j", "8"]
    for p in SEMGREP_PACKS:
        cmd += ["--config", p]
    out = subprocess.run(cmd + [str(tree)], capture_output=True, text=True)
    d = json.loads(out.stdout)
    if d.get("errors") and any(e.get("level") == "error" and "download" in e.get("message", "") for e in d["errors"]):
        raise RuntimeError(d["errors"])
    print("semgrep scanned", len(d.get("paths", {}).get("scanned", [])), "files,", len(d["results"]), "findings")
    res = defaultdict(list)
    for r in d["results"]:
        res[str(Path(r["path"]).resolve().relative_to(tree))].append((r["check_id"], r["extra"]["severity"]))
    return res


def bandit(tree):
    out = subprocess.run(["bandit", "-q", "-r", "-f", "json", str(tree)], capture_output=True, text=True)
    d = json.loads(out.stdout)
    res = defaultdict(list)
    for r in d["results"]:
        res[str(Path(r["filename"]).resolve().relative_to(tree))].append((r["test_id"], r["issue_severity"]))
    return res


def main(dataset):
    src = {"heval": "heval_fix.jsonl", "swe": "swe_patches.jsonl"}[dataset]
    items = [json.loads(l) for l in open(ROOT / "bench" / "data" / src)]
    with tempfile.TemporaryDirectory() as d:
        tree = Path(d).resolve()
        files = {}  # item id -> list of (pre_rel or None, post_rel)
        for k, it in enumerate(items):
            if dataset == "heval":
                rel = f"h{k}/code.py"
                (tree / rel).parent.mkdir(parents=True, exist_ok=True)
                (tree / rel).write_text(it["code"])
                files[it["id"]] = [(None, rel)]
            else:
                pairs = []
                for j, (path, post) in enumerate((it["files_post"] or {}).items()):
                    pre = it["files_pre"].get(path, "")
                    pr, po = f"s{k}/pre/f{j}.py", f"s{k}/post/f{j}.py"
                    for rel, txt in ((pr, pre), (po, post)):
                        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
                        (tree / rel).write_text(txt)
                    pairs.append((pr, po))
                files[it["id"]] = pairs
        tools = {"semgrep": semgrep(tree), "bandit": bandit(tree)}
    out = ROOT / "results" / "rebuild" / f"sast_{dataset}.jsonl"
    with open(out, "w") as f:
        for it in items:
            rec = {"id": it["id"], "label": it["label"], "pair": it["pair"], "generator": it.get("generator")}
            for name, res in tools.items():
                new = Counter()
                for pre, post in files[it["id"]]:
                    c_post = Counter(res.get(post, []))
                    c_pre = Counter(res.get(pre, [])) if pre else Counter()
                    new += c_post - c_pre
                rec[name] = {"findings": [[r, s, n] for (r, s), n in new.items()],
                             "score": sum(W.get(s.upper(), 0.3) * n for (r, s), n in new.items())}
            f.write(json.dumps(rec) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
