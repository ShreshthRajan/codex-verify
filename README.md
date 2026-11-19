# CodeX-Verify: an execution-grounded re-evaluation of multi-agent code verification

CodeX-Verify is a four-agent verifier for Python code: rule-based Correctness, Security,
Performance, and Style agents whose findings an aggregator turns into PASS / WARNING / FAIL.
This repository holds the system (`src/`), an execution-labeled benchmark (`bench/`), every
verifier's raw outputs (`results/rebuild/`), and the paper (`paper/`).

## Main result

On labels that come from running tests, the static multi-agent verifier is at chance, and
splitting an LLM judge into specialist agents does not help.

| Verifier | HumanEvalFix AUC | SWE-bench Verified agent patches AUC | Merged human fixes flagged |
|---|---|---|---|
| CodeX-Verify (4 static agents) | 0.50 | 0.48 | 100% |
| Semgrep / Bandit | 0.50 / 0.50 | 0.52 / 0.48 | 2% / 2% |
| Claude Sonnet 5.5, single judge | 1.00 | 0.89 | 4.7% |
| Claude Sonnet 5.5, 4 specialist agents (mean) | 0.99 | 0.88 | 1.3% |

The numbers are produced by `bench/analyze.py` (see `paper/tables/main.tex`). The paper also
measures agent dependence and re-examines the earlier 99-sample evaluation, whose labels were
generated together with the samples and whose 68.7% accuracy is below the 71.7% majority
baseline.

Note: the Correctness agent executes the code it is given (`_safe_execution_check` in
`src/agents/correctness_critic.py`). Run it only on code you trust, or in a sandbox.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install anthropic datasets semgrep bandit scikit-learn scipy numpy matplotlib pyyaml aiohttp python-dotenv
python bench/build_heval.py          # 164 HumanEvalFix pairs, labels re-verified by running the tests
python bench/build_swe.py            # 150 SWE-bench Verified tasks x 2 agents + gold patches
python bench/run_static.py heval && python bench/run_static.py swe
python bench/run_sast.py heval && python bench/run_sast.py swe
python bench/run_llm.py claude-sonnet-5-5 swe single,correctness,security,performance,style   # needs ANTHROPIC_API_KEY
python bench/analyze.py              # tables, figures, and in-text numbers for the paper
python bench/old_set_leak.py         # re-analysis of the earlier 99-sample set
```

Committed outputs in `results/rebuild/` let `bench/analyze.py` run without any API calls.

## Layout

- `src/` — the CodeX-Verify agents, orchestrator, and aggregator (unchanged from the earlier evaluation).
- `bench/` — benchmark builders, verifier runners, and the analysis.
- `results/rebuild/` — raw outputs of every verifier on every item, and `summary.json`.
- `paper/` — the paper; `paper/archive/` and `docs/archive/` hold superseded drafts.
- `experiments/`, `results/` (other than `rebuild/`) — scripts and outputs of the earlier evaluation, kept for the re-analysis.
