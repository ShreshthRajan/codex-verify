# CodeX-Verify: Multi-Agent Code Verification Framework

**Enterprise-grade verification system addressing the 40–60% false positive rate in LLM code generation (Codex, SWE-bench, GPT-Code).**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Verification Accuracy](https://img.shields.io/badge/verification-70.6%25-green.svg)](docs/evaluation.md)

---

## Current State 

LLM-generated code is  unsuitable for direct enterprise deployment due to a **40–60% rate of undetected bad outputs**, as demonstrated by SWE-bench, Meta Prompt Testing, and SecRepoBench. Code may pass test cases but still fails in production due to subtle correctness, security, performance, or maintainability issues.

The primary goal of this project is to improve **true positive detection** of such bad outputs — reducing the risk of unsafe code shipping undetected. False positive rate on “good”* code is tunable, but not the immediate focus.

**CodeX‑Verify** is a *multi-agent verification* system addressing this barrier:

* **70.6% accuracy** (+30.6 pts over Codex baseline)
* **91.7% true positive rate** for actual bugs
* **98% detection** of SWE-bench real production issues
* **Sub‑200 ms latency** (CI/CD and PR pipeline ready)
* **Enterprise production gating** with zero-tolerance deploy blockers


*“good code” = should pass verification without blocking deploy — correct, secure, performant, maintainable


---

## Architecture Overview

```mermaid
graph TB
    Input[Code Input] --> Orchestrator[AsyncOrchestrator]
    Orchestrator --> |Parallel Execution| A1[CorrectnessAgent]
    Orchestrator --> |Parallel Execution| A2[SecurityAuditor] 
    Orchestrator --> |Parallel Execution| A3[PerformanceProfiler]
    Orchestrator --> |Parallel Execution| A4[StyleMaintainabilityJudge]
    
    A1 --> Aggregator[ResultAggregator]
    A2 --> Aggregator
    A3 --> Aggregator
    A4 --> Aggregator
    
    Aggregator --> |Enterprise Scoring| Report[VerificationReport]
    Aggregator --> |Compound Detection| Vulnerabilities[CompoundVulnerabilities]
    
    Report --> Dashboard[StreamlitDashboard]
    Report --> CLI[CLIInterface]
```
---

## Core Innovations: Multi-Agent Verification

### Correctness Critic

* AST-based static analysis (Python, etc.)
* Exception path analysis
* Input validation & type safety
* Edge case coverage
* Semantic contract validation

### Security Auditor

* Compound vulnerability detection with exponential risk scoring
* OWASP Top 10 / CWE mapping
* Entropy-based secret detection
* Context-aware severity escalation
* Code execution path & injection vulnerabilities

### Performance Profiler

* Context-aware algorithmic complexity analysis
* Scale-aware performance thresholds
* Memory/resource profiling (leak detection)
* Bottleneck identification with optimization suggestions

### Style & Maintainability Judge

* Multi-linter integration (Black, Flake8, Pylint)
* Halstead complexity & technical debt
* Documentation coverage analysis
* Architectural pattern (SOLID) checks
* Code smell detection & refactoring recommendations

---

---


## Technical Breakthroughs

1. **Compound Vulnerability Detection** — Novel compound vulnerability detection with risk amplification scoring:

```python
compound_multipliers = {
    ('sql_injection','hardcoded_secret'):3.0,
    ('code_execution','dangerous_import'):2.0,
    ('complexity','algorithm_inefficiency'):1.8
}
```

2. **Context-Aware Analysis** — thresholds adapted for patch vs snippet vs full file

3. **Enterprise Production Scoring** — zero-tolerance deploy gating:

```python
if critical_count>0 or compound_vulns:
    return "FAIL"
```

4. **Parallel Agent Orchestration** — sub‑200 ms async execution:

```python
async def _execute_agents_parallel(self, code, context):
    tasks = {name: asyncio.create_task(agent.analyze(code, context)) 
             for name, agent in self.agents.items()}
    return await asyncio.gather(*tasks.values())
```

5. **Local-First Architecture** — full offline operation for enterprise privacy needs

---

## Comparison to SOTA

| System                     | Accuracy/Performance | False Positive Rate | True Positive Rate | Scope/Notes                                    |
|----------------------------|---------------------|-------------------|-------------------|-----------------------------------------------|
| **o3 (SWE-bench Verified)**   | 72% (2025)         | —                 | —                 | Latest OpenAI reasoning model                  |
| **GPT-4.1 (SWE-bench Verified)** | 54.6% (2025)    | —                 | —                 | Latest OpenAI API model                       |
| **Meta Prompt Testing**       | 89.0% (boosted)    | 8.6%             | 75%               | Function-level validation (Wang & Zhu, 2024) |
| **SecRepoBench**              | <25% secure-pass@1 | —                | —                 | Repository-level security (318 tasks, 19 LLMs)|
| **BaxBench**                  | 38% secure-pass@1  | —                | —                 | Backend security (392 tasks, GPT-4 best)     |
| **Static Analyzers**          | ~65%               | 15-25%           | 60-80%            | Traditional SAST tools                        |
| **SWE-bench Empirical Study** | 29.6% incorrect    | —                | 98% detection     | Real production bugs (Xia et al., 2025)      |
| **CodeX-Verify (ours)**       | **70.6%**          | **80%** (tunable)| **91.7%**         | **Full multi-agent across all dimensions**   |

*Note:*
FPR is currently high on "good code" because the verifier is enforced to be **strict for enterprise deployment**:

```yaml
max_critical_vulnerabilities: 0
max_high_vulnerabilities: 1
crypto_compliance_required: True
```

**Any flagged issue blocks deploy**, as intended.
This FPR is tunable but not the focus of the SWE-bench goal (which targets reducing *false positives on bad Codex outputs*).
The verifier exceeds that goal with 91.7% TPR and 98% real-world detection.

---

## **Relevant Papers**

1. *Validating LLM-Generated Programs with Metamorphic Prompt Testing* - Wang & Zhu (2024)
2. *Are "Solved Issues" in SWE-bench Really Solved Correctly? An Empirical Study* - Xia et al. (2025)
3. *SecRepoBench: Benchmarking LLMs for Secure Code Generation in Real-World Repositories* - Dilgren et al. (2025)
4. *BaxBench: Can LLMs Generate Secure and Correct Backends?* - Vero et al. (2025)
5. *Vulnerability Detection with Code Language Models: How Far Are We?* - Ding et al. (2024)
6. *When LLMs meet cybersecurity: a systematic literature review* - (2025)
7. *SWE-bench: Can Language Models Resolve Real-world Github Issues?* - Jimenez et al. (2024)
8. *Utilizing Precise and Complete Code Context to Guide LLM in Automatic False Positive Mitigation* - (2024)
9. *Minimizing False Positives in Static Bug Detection via LLM-Enhanced Path Feasibility Analysis* - (2024)
10. *LLM4CodeBench*
11. *SWE‑bench: Can Foundation Models Solve Software Engineering Tasks?*

---

## Performance Validation

### Comprehensive Test Results (34 cases)

```
✅ Accuracy: 70.6%  
🎯 TPR: 91.7%  
⚠️ FPR: 80% (good code — tuning in progress)

Baseline:  
Codex: ~40%  
Static Analyzers: ~65%  
CodeX‑Verify: +30.6% over Codex
```

### Breakdown by Category

* Algorithmic Complexity: **100%**
* Resource Management: **100%**
* Scalability Performance: **100%**
* Edge Case Logic: **100%**
* Security Validation: **83.3%**
* Input Validation: **66.7%**

---

### Real-World SWE-bench (50 samples)

* **98% detection** of production bugs in live open-source projects
* Targets: `django`, `requests`, `pytorch`, `scikit-learn`, `pandas`

---































## Enterprise Features

```python
enterprise_thresholds = {
    'max_critical_vulnerabilities': 0,
    'max_high_vulnerabilities': 1,
    'max_secrets_per_file': 0,
    'crypto_compliance_required': True
}
```

* Risk scoring: LOW / MEDIUM / HIGH / CRITICAL
* Compound risk detection
* Multi-tier caching (<200 ms)
* Horizontal scaling (stateless agents)
* Offline mode for air-gapped environments
* Integration-ready: GitHub Actions, Jenkins, SonarQube, SIEM

---

## Quick Start

```bash
git clone https://github.com/your‑org/codex‑verify.git
cd codex‑verify
pip install ‑r requirements.txt

python swe_bench_mirror_evaluator.py
python swe_bench_real_evaluator.py

streamlit run ui/streamlit_dashboard.py
```

---

## Project Structure

## Project Structure

```text
codex-verify/
├── src/
│   ├── agents/                 # 4 verification agents
│   │   ├── base_agent.py
│   │   ├── correctness_critic.py
│   │   ├── security_auditor.py
│   │   ├── performance_profiler.py
│   │   ├── style_maintainability_judge.py
│   ├── orchestration/          # Async orchestration engine
│   │   ├── async_orchestrator.py
│   │   ├── caching_layer.py
│   │   ├── result_aggregator.py
├── ui/
│   ├── cli_interface.py        # CLI interface
│   ├── streamlit_dashboard.py  # Streamlit dashboard
│   ├── components/             # UI components
│   │   ├── code_editor.py
│   │   ├── feedback_component.py
│   │   ├── metrics_charts.py
│   │   ├── verification_report.py
├── config/                     # YAML config files
├── tests/                      # Comprehensive test suite
├── demo_samples/               # Demo test cases & example data
├── swe_bench_real_evaluator.py # Real SWE-bench runner
├── swe_bench_mirror_evaluator.py # Mirror benchmark runner
├── README.md
├── requirements.txt
├── setup.py
```


## Development & Testing

```bash
python -m pytest tests/ -v
python swe_bench_real_evaluator.py
```

---

## Conclusion

**CodeX‑Verify** is a full *multi-agent verification system* designed to close the 40–60% false positive gap in Codex and other LLM-generated code, validated across SWE‑bench and real-world production issues.
