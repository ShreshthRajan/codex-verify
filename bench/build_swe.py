"""Build the SWE-bench Verified patch-verification set.

Agent patches come from two public leaderboard submissions; each patch's label
is the official SWE-bench harness verdict recorded in the submission's
results.json (resolved = FAIL_TO_PASS and PASS_TO_PASS tests all pass).
For every patch we also fetch the touched Python files at the task's base
commit and apply the patch, so static tools can analyse real pre/post files.
"""
import json, random, subprocess, tempfile, urllib.request, base64, os, re, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "data" / "swe_patches.jsonl"
CACHE = RAW / "files"
GENERATORS = ["20241022_tools_claude-3-5-sonnet-updated", "20241028_agentless-1.5_gpt4o"]
N_TASKS = 150
SEED = 0


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def gh_json(path):
    out = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=120)
    return json.loads(out.stdout)


def load_generator(g):
    pf = RAW / f"{g}_all_preds.jsonl"
    if not pf.exists():
        pf.write_bytes(fetch(f"https://swe-bench-submissions.s3.amazonaws.com/verified/{g}/all_preds.jsonl"))
    rf = RAW / f"{g}_results.json"
    if not rf.exists():
        c = gh_json(f"repos/SWE-bench/experiments/contents/evaluation/verified/{g}/results/results.json")
        rf.write_bytes(base64.b64decode(c["content"]))
    preds = {}
    for line in open(pf):
        p = json.loads(line)
        preds[p["instance_id"]] = p.get("model_patch") or ""
    resolved = set(json.load(open(rf))["resolved"])
    return preds, resolved


def touched_files(patch):
    return [m for m in re.findall(r"^diff --git a/(\S+) b/", patch, re.M)]


def python_sections(patch):
    """Keep only the diff sections that touch .py files (drops binary/doc/config hunks)."""
    parts = re.split(r"(?m)^(?=diff --git )", patch)
    keep = [p for p in parts if p.startswith("diff --git") and
            re.match(r"diff --git a/\S+\.py b/", p)]
    return "".join(keep)


def base_file(repo, commit, path):
    dst = CACHE / repo.replace("/", "__") / commit / path
    if dst.exists():
        return dst.read_text(errors="replace")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = fetch(f"https://raw.githubusercontent.com/{repo}/{commit}/{path}").decode(errors="replace")
    except Exception:
        data = ""  # new file
    dst.write_text(data)
    return data


def apply_patch(files, patch):
    """files: {path: text} at base. Returns {path: post text} or None if patch fails."""
    with tempfile.TemporaryDirectory() as d:
        for p, t in files.items():
            fp = Path(d) / p
            fp.parent.mkdir(parents=True, exist_ok=True)
            if t:
                fp.write_text(t)
        subprocess.run(["git", "init", "-q"], cwd=d)
        r = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], cwd=d, input=patch,
                           text=True, capture_output=True)
        if r.returncode != 0:
            return None
        out = {}
        for p in files:
            fp = Path(d) / p
            out[p] = fp.read_text(errors="replace") if fp.exists() else ""
        return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(RAW / "swebench_verified.jsonl")}
    gens = {g: load_generator(g) for g in GENERATORS}
    ids = sorted(tasks)
    random.Random(SEED).shuffle(ids)
    # keep tasks where every generator produced a non-empty patch
    chosen = [i for i in ids if all(gens[g][0].get(i, "").strip() for g in GENERATORS)][:N_TASKS]

    jobs = []
    for iid in chosen:
        t = tasks[iid]
        for g in GENERATORS:
            jobs.append((iid, g, gens[g][0][iid], int(iid not in gens[g][1])))
        jobs.append((iid, "gold", t["patch"], 0))

    def build(job):
        iid, g, patch, label = job
        t = tasks[iid]
        py_patch = python_sections(patch)
        paths = touched_files(py_patch)
        base = {p: base_file(t["repo"], t["base_commit"], p) for p in paths}
        post = apply_patch(base, py_patch) if paths else {}
        return {
            "id": f"{iid}__{g}", "pair": iid, "dataset": "swe_verified", "generator": g,
            "label": label, "repo": t["repo"], "issue": t["problem_statement"],
            "patch": patch, "files_pre": base, "files_post": post,
            "patch_applies": post is not None,
        }

    with ThreadPoolExecutor(16) as ex:
        items = list(ex.map(build, jobs))
    with open(OUT, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    from collections import Counter
    print(Counter((it["generator"], it["label"]) for it in items))
    print("patch fails to apply:", sum(not it["patch_applies"] for it in items))


if __name__ == "__main__":
    main()
