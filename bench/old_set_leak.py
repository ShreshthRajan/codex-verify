"""Re-examine the original 99-sample evaluation set (v1 of the paper).

1. Reconstruct the 99 samples exactly as experiments/evaluate_combined_100_samples.py
   built them (29 unique hand-written mirror samples + the 70 Claude-generated
   samples in results/claude_patches/claude_generated_66_samples_20251024_113425.json).
2. Count label-leaking comments (# BUG, # VULNERABLE, ...).
3. Re-run CodeX-Verify on the original and on comment-stripped code.
4. Run the Sonnet 5.5 single judge on original and comment-stripped code.
Output: results/rebuild/old_set_leak.json
"""
import asyncio, io, json, re, sys, tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))
from swe_bench_mirror_evaluator import create_comprehensive_samples  # noqa: E402
from src.orchestration.async_orchestrator import AsyncOrchestrator, VerificationConfig  # noqa: E402
import run_llm  # noqa: E402

LEAK = re.compile(r"#.*\b(BUG|VULN\w*|INSECURE|WRONG|FIX(ME)?|ISSUE|PROBLEM|LEAK|UNSAFE|DANGER\w*|INJECTION|RACE)\b", re.I)


def strip_comments(code):
    try:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type != tokenize.COMMENT]
        return tokenize.untokenize(toks)
    except Exception:
        return "\n".join(re.sub(r"\s#.*$", "", l) for l in code.splitlines())


def samples():
    seen, out = set(), []
    for s in create_comprehensive_samples():
        if s.problem_id not in seen:
            seen.add(s.problem_id)
            out.append({"id": s.problem_id, "source": "mirror", "code": s.codex_solution,
                        "label": int(s.should_be_rejected)})
    for k, s in enumerate(json.load(open(ROOT / "results/claude_patches/claude_generated_66_samples_20251024_113425.json"))):
        out.append({"id": f"{s['problem_id']}#{k}", "source": "claude", "code": s["codex_solution"],
                    "label": int(s["should_be_rejected"])})
    return out


def metrics(y, f):
    tp = sum(a and b for a, b in zip(f, y)); tn = sum((not a) and (not b) for a, b in zip(f, y))
    P = sum(y); N = len(y) - P
    return {"acc": (tp + tn) / len(y), "tpr": tp / P, "fpr": (N - tn) / N, "tp": tp, "tn": tn, "n": len(y)}


async def llm_all(items, key):
    import anthropic
    client = anthropic.AsyncAnthropic(max_retries=4); sem = asyncio.Semaphore(16)
    jobs = []
    for it in items:
        q = {"id": it["id"], "dataset": "heval_fix", "code": it[key]}
        jobs.append(run_llm.call(client, sem, "claude-sonnet-5-5", q, "single"))
    return await asyncio.gather(*jobs)


def main():
    items = samples()
    for it in items:
        it["stripped"] = strip_comments(it["code"])
        it["leaky_comments"] = len(LEAK.findall(it["code"]))
        it["bug_comment"] = bool(re.search(r"#\s*BUG", it["code"]))
    y = [it["label"] for it in items]
    res = {"n": len(items), "buggy": sum(y), "majority_baseline_acc": max(sum(y), len(y) - sum(y)) / len(y),
           "samples_with_#BUG": sum(it["bug_comment"] for it in items),
           "samples_with_any_leaky_comment": sum(it["leaky_comments"] > 0 for it in items),
           "buggy_with_#BUG": sum(it["bug_comment"] and it["label"] for it in items),
           "correct_with_#BUG": sum(it["bug_comment"] and not it["label"] for it in items)}
    for key in ("code", "stripped"):
        flags, fails = [], []
        orch = AsyncOrchestrator(VerificationConfig.default())
        for it in items:
            rep = asyncio.run(orch.verify_code(it[key], {}))
            flags.append(rep.overall_status != "PASS")
            fails.append(rep.overall_status == "FAIL")
        res[f"codexverify_{key}"] = metrics(y, flags)
        res[f"codexverify_FAILonly_{key}"] = metrics(y, fails)
        out = asyncio.run(llm_all(items, key))
        lf = [o.get("verdict") == "REJECT" for o in out]
        res[f"sonnet_single_{key}"] = metrics(y, lf)
        res[f"sonnet_single_{key}_errors"] = sum("error" in o for o in out)
    (ROOT / "results/rebuild/old_set_leak.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
