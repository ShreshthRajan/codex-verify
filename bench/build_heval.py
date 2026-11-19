"""Build the HumanEvalFix-Python verification set (164 buggy + 164 correct).

Each item is a full function: the HumanEval prompt (signature + docstring)
followed by either the canonical or the human-injected buggy body.
Labels are execution-grounded: we run every item against the HumanEval tests
and keep only pairs where the correct version passes and the buggy one fails.
"""
import json, subprocess, sys, tempfile, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw" / "humanevalpack_python.jsonl"
OUT = ROOT / "data" / "heval_fix.jsonl"


def run_tests(code: str, test: str, entry: str, timeout=10) -> bool:
    prog = code + "\n\n" + test + f"\n\ncheck({entry})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
        path = f.name
    try:
        r = subprocess.run([sys.executable, "-I", path], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    items, dropped = [], []
    for line in open(RAW):
        r = json.loads(line)
        imp = r["import"] or ""
        good = imp + r["prompt"] + r["canonical_solution"]
        bad = imp + r["prompt"] + r["buggy_solution"]
        ok_good = run_tests(good, r["test"], r["entry_point"])
        ok_bad = run_tests(bad, r["test"], r["entry_point"])
        if not ok_good or ok_bad:
            dropped.append((r["task_id"], ok_good, ok_bad))
            continue
        tid = r["task_id"].replace("/", "_")
        for code, label, tag in ((good, 0, "fixed"), (bad, 1, "buggy")):
            items.append({
                "id": f"{tid}_{tag}", "pair": tid, "dataset": "heval_fix",
                "label": label, "code": code, "spec": r["prompt"],
                "bug_type": r["bug_type"] if label else None,
                "tests_pass": not label,
            })
    with open(OUT, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"kept {len(items)} items ({len(items)//2} pairs); dropped {dropped}")


if __name__ == "__main__":
    main()
