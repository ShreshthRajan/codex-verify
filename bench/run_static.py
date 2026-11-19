"""Run the CodeX-Verify static agents on the benchmark.

Each of the four agents runs once per input; the 15 agent subsets are then
formed by passing the corresponding subset of agent results to the system's
own ResultAggregator (identical to running the orchestrator with only those
agents enabled, without re-running the agents 15 times).

Inputs:
  heval_fix : the full function (signature + docstring + body).
  swe       : two views of each patch.
    snippet - the added + context lines of the diff, with the
              swe_bench_sample context flag (the original paper's mode).
    diff    - every touched .py file before and after the patch; the score is
              the severity-weighted count of issues present after but not
              before (a patch-aware variant that gives static checks their
              fairest chance).
Output: results/rebuild/static_<dataset>.jsonl
"""
import asyncio, json, sys, time
from collections import Counter
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from agents.base_agent import AgentResult
from agents.correctness_critic import CorrectnessAgent
from agents.security_auditor import SecurityAuditor
from agents.performance_profiler import PerformanceProfiler
from agents.style_maintainability import StyleMaintainabilityJudge
from orchestration.async_orchestrator import VerificationConfig
from orchestration.result_aggregator import ResultAggregator

AGENTS = ["correctness", "security", "performance", "style"]
SUBSETS = [c for k in range(1, 5) for c in combinations(AGENTS, k)]
SEV_W = {"LOW": 0.1, "MEDIUM": 0.3, "HIGH": 0.7, "CRITICAL": 1.0}

_agents = None


def agents():
    global _agents
    if _agents is None:
        _agents = {"correctness": CorrectnessAgent(), "security": SecurityAuditor(),
                   "performance": PerformanceProfiler(), "style": StyleMaintainabilityJudge()}
    return _agents


def sev(issue):
    s = issue.severity
    return getattr(s, "name", str(s)).upper()


async def run_agents(code, ctx):
    res = {}
    for name, a in agents().items():
        try:
            res[name] = await asyncio.wait_for(a.analyze(code, dict(ctx)), timeout=60)
        except Exception as e:  # mirrors orchestrator: failed agent -> score 0
            res[name] = AgentResult(agent_name=name, execution_time=0.0, overall_score=0.0,
                                    issues=[], metadata={"error": str(e)}, success=False,
                                    error_message=str(e))
    return res


async def subset_reports(agent_results, code, ctx):
    agg = ResultAggregator(VerificationConfig.default())
    out = {}
    for s in SUBSETS:
        sub = {k: agent_results[k] for k in s}
        try:
            rep = await agg.aggregate_results(sub, code, ctx, time.time())
            out["+".join(s)] = {"score": rep.overall_score, "status": rep.overall_status}
        except Exception as e:  # the orchestrator turns aggregator crashes into an ERROR report, score 0
            out["+".join(s)] = {"score": 0.0, "status": "ERROR", "error": repr(e)}
    return out


def agent_summary(agent_results):
    return {k: {"score": r.overall_score, "success": r.success,
                "issues": [[i.type, sev(i)] for i in r.issues]} for k, r in agent_results.items()}


MAX_SNIPPET = 30000  # same cap as the LLM judges' patch input


def python_sections(patch):
    import re
    parts = re.split(r"(?m)^(?=diff --git )", patch)
    return "".join(p for p in parts if re.match(r"diff --git a/\S+\.py b/", p))


def extract_snippet(patch):
    """Same extraction as experiments/swe_bench_real_evaluator.py (original paper)."""
    lines, out, in_code = patch.splitlines(), [], False
    for line in lines:
        if line.startswith(("diff --git", "index ", "+++", "---")):
            continue
        if line.startswith("@@"):
            in_code = True
            continue
        if in_code:
            if line.startswith("+"):
                out.append(line[1:])
            elif line.startswith(" "):
                out.append(line[1:])
    code = "\n".join(out)
    return patch if len(code.strip()) < 50 else code


def process(item):
    t0 = time.time()
    rec = {"id": item["id"], "label": item["label"], "dataset": item["dataset"],
           "pair": item["pair"], "generator": item.get("generator")}
    if item["dataset"] == "heval_fix":
        ar = asyncio.run(run_agents(item["code"], {}))
        rec["agents"] = agent_summary(ar)
        rec["subsets"] = asyncio.run(subset_reports(ar, item["code"], {}))
    else:
        code = extract_snippet(python_sections(item["patch"]) or item["patch"])[:MAX_SNIPPET]
        ctx = {"repo": item["repo"], "github_issue": True, "real_world_code": True,
               "swe_bench_sample": True}
        ar = asyncio.run(run_agents(code, ctx))
        rec["agents"] = agent_summary(ar)
        rec["subsets"] = asyncio.run(subset_reports(ar, code, ctx))
        # patch-aware view: new issues per agent across touched files
        new = {a: Counter() for a in AGENTS}
        for path, post in (item["files_post"] or {}).items():
            pre = item["files_pre"].get(path, "")
            r_post = asyncio.run(run_agents(post, {})) if post.strip() else {}
            r_pre = asyncio.run(run_agents(pre, {})) if pre.strip() else {}
            for a in AGENTS:
                c_post = Counter((i.type, sev(i)) for i in (r_post[a].issues if a in r_post else []))
                c_pre = Counter((i.type, sev(i)) for i in (r_pre[a].issues if a in r_pre else []))
                new[a] += (c_post - c_pre)
        rec["diff_new_issues"] = {a: [[t, s, n] for (t, s), n in c.items()] for a, c in new.items()}
    rec["seconds"] = time.time() - t0
    return rec


def main(dataset):
    src = {"heval": "heval_fix.jsonl", "swe": "swe_patches.jsonl"}[dataset]
    items = [json.loads(l) for l in open(ROOT / "bench" / "data" / src)]
    out = ROOT / "results" / "rebuild" / f"static_{dataset}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in open(out)}
    items = [it for it in items if it["id"] not in done]
    print(len(done), "already done;", len(items), "to run", flush=True)
    with Pool(8) as p, open(out, "a") as f:
        for i, rec in enumerate(p.imap_unordered(process, items, chunksize=1)):
            f.write(json.dumps(rec) + "\n")
            if i % 25 == 0:
                print(i, rec["id"], f"{rec['seconds']:.2f}s", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
