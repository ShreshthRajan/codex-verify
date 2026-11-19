"""LLM verifiers on the benchmark: one generalist judge and four specialist
agents (correctness, security, performance, style) that mirror CodeX-Verify's
roles. Every call returns a probability that the code is defective, which
gives each verifier a full ROC/PR curve.

usage: python bench/run_llm.py <model> <dataset: heval|swe> <roles: single,correctness,...>
Results are cached in results/rebuild/llm_<model>_<dataset>.jsonl (one line per
item x role); reruns only make the missing calls.
"""
import asyncio, json, os, sys, time
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
PRICE = {"claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50),
         "claude-opus-5-5": (4.0, 20.0)}
MAX_PATCH = 30000

FOCUS = {
    "single": "Judge overall correctness: does the code do what it is supposed to do, for all valid inputs, without introducing problems?",
    "correctness": "Focus only on functional correctness: logic errors, wrong results, off-by-one errors, unhandled edge cases, wrong exception behaviour.",
    "security": "Focus only on security: injection, unsafe deserialization, secrets, path traversal, unsafe use of eval/exec/subprocess, and other vulnerabilities.",
    "performance": "Focus only on performance: algorithmic complexity, unnecessary work, resource leaks, scalability problems.",
    "style": "Focus only on maintainability and style: readability, naming, structure, documentation, code smells.",
}

SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "p_defect": {"type": "integer"},
        "verdict": {"type": "string", "enum": ["ACCEPT", "REJECT"]},
    },
    "required": ["reason", "p_defect", "verdict"],
    "additionalProperties": False,
}

SYSTEM = ("You are an expert Python code reviewer acting as a pre-merge verifier. "
          "You cannot run the code. Answer with the JSON object requested.")


def prompt(item, role):
    focus = FOCUS[role]
    tail = ("\n\nReturn JSON with: reason (one or two sentences), p_defect (integer 0-100: "
            "your probability that the code should be rejected because of a defect in your area of focus), "
            "verdict (REJECT if it should not be merged, otherwise ACCEPT).")
    if item["dataset"] == "heval_fix":
        return (f"{focus}\n\nThe function below has a docstring that specifies its intended behaviour. "
                f"Decide whether the implementation is acceptable.\n\n```python\n{item['code']}\n```" + tail)
    patch = item["patch"]
    if len(patch) > MAX_PATCH:
        patch = patch[:MAX_PATCH] + f"\n[... patch truncated, {len(item['patch']) - MAX_PATCH} more characters ...]"
    return (f"{focus}\n\nA developer submitted the patch below to resolve this GitHub issue in {item['repo']}. "
            f"Decide whether the patch is acceptable: it should resolve the issue and not break existing behaviour "
            f"(the repository's hidden tests will be run on it).\n\n<issue>\n{item['issue']}\n</issue>\n\n"
            f"<patch>\n{patch}\n</patch>" + tail)


async def call(client, sem, model, item, role):
    async with sem:
        for attempt in range(6):
            try:
                r = await client.messages.create(
                    model=model, max_tokens=4000, system=SYSTEM,
                    output_config={"effort": "low",
                                   "format": {"type": "json_schema", "schema": SCHEMA}},
                    messages=[{"role": "user", "content": prompt(item, role)}])
                text = next((b.text for b in r.content if b.type == "text"), "")
                if r.stop_reason == "refusal" or not text:
                    out = {"error": f"stop_reason={r.stop_reason}"}
                else:
                    out = json.loads(text)
                return {"id": item["id"], "role": role, "model": model, **out,
                        "in_tok": r.usage.input_tokens, "out_tok": r.usage.output_tokens}
            except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError,
                    json.JSONDecodeError) as e:
                await asyncio.sleep(2 ** attempt)
                err = str(e)
        return {"id": item["id"], "role": role, "model": model, "error": err}


async def main(model, dataset, roles):
    src = {"heval": "heval_fix.jsonl", "swe": "swe_patches.jsonl"}[dataset]
    items = [json.loads(l) for l in open(ROOT / "bench" / "data" / src)]
    out = ROOT / "results" / "rebuild" / f"llm_{model}_{dataset}.jsonl"
    done = set()
    if out.exists():
        for l in open(out):
            r = json.loads(l)
            if "error" not in r:
                done.add((r["id"], r["role"]))
    todo = [(it, ro) for ro in roles for it in items if (it["id"], ro) not in done]
    print(f"{len(todo)} calls to make", flush=True)
    client = anthropic.AsyncAnthropic(max_retries=4)
    sem = asyncio.Semaphore(int(os.environ.get("CONC", "16")))
    pin, pout = PRICE[model]
    cost, n = 0.0, 0
    with open(out, "a") as f:
        for fut in asyncio.as_completed([call(client, sem, model, it, ro) for it, ro in todo]):
            r = await fut
            f.write(json.dumps(r) + "\n"); f.flush()
            cost += (r.get("in_tok", 0) * pin + r.get("out_tok", 0) * pout) / 1e6
            n += 1
            if n % 100 == 0:
                print(f"{n}/{len(todo)} cost so far ${cost:.2f}", flush=True)
    print(f"done {n} calls, ${cost:.2f}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3].split(",")))
