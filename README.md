# CodeX-Verify: Multi-Agent Code Verification Framework

**Multi-agent verification system for detecting bugs in LLM-generated code with information-theoretic foundations.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv-red.svg)](https://arxiv.org/abs/XXXXX)

---

## Overview

Large language models generate code with 40-60% latent bug rates. CodeX-Verify addresses this through multi-agent verification across four orthogonal dimensions: correctness, security, performance, and maintainability.

**Key Results (99-sample evaluation with perfect ground truth):**
* **68.7% accuracy** (±9.1% CI) - improves 28.7pp over Codex baseline (40%)
* **76.1% TPR** - matches SOTA Meta Prompt Testing (75%)
* **50.0% FPR** - reflects static analysis vs test execution tradeoff
* **+39.7pp multi-agent advantage** - validated through 15-configuration ablation
* **Sub-200ms latency** - CI/CD ready

---

## Novel Contributions

### 1. Information-Theoretic Foundations
Formal proof that multi-agent systems achieve higher mutual information with bug presence: $I(A_1, A_2, A_3, A_4; B) > \max_i I(A_i; B)$ when agents exhibit low correlation (measured ρ = 0.05-0.25).

### 2. Compound Vulnerability Detection
First formalization of exponential risk amplification for co-occurring vulnerabilities:
```python
Risk(v1 ∪ v2) = Risk(v1) × Risk(v2) × α(v1, v2)
# α ∈ {1.5, 2.0, 2.5, 3.0} calibrated from attack chains
# Example: SQL injection + credentials = 15× impact (300 vs 20)
```

### 3. Comprehensive Ablation Validation
Systematic testing across 15 agent configurations proves:
- Single-agent average: 32.8%
- 4-agent system: 72.4%
- Improvement: +39.7pp (exceeds AutoReview's +18.72% F1 by 2×)
- Diminishing returns: +14.9pp, +13.5pp, +11.2pp for agents 2, 3, 4

---

## Architecture

Four specialized agents analyze code in parallel:

### Correctness Critic (Solo: 75.9%)
- AST analysis (cyclomatic complexity, nesting depth)
- Exception path analysis (80% coverage target)
- Input validation detection (70% target)
- Edge case coverage
- Contract validation (docstring vs implementation)

### Security Auditor (Solo: 20.7%)
- 15+ vulnerability patterns (SQL injection, code execution, deserialization)
- Entropy-based secret detection (H > 3.5)
- CWE/OWASP mapping
- **Compound vulnerability detection** (novel)

### Performance Profiler (Solo: 17.2%)
- Algorithm complexity classification (O(1) through O(2^n))
- Context-aware thresholds (patch vs full file)
- Bottleneck identification
- Resource leak detection

### Style & Maintainability (Solo: 17.2%)
- Multi-linter integration (Black, Flake8, Pylint)
- Halstead complexity, maintainability index
- Documentation coverage
- **All issues LOW severity** (never blocks deployment)

---

## Installation

```bash
git clone https://github.com/ShreshthRajan/codex-verify.git
cd codex-verify
pip install -r requirements.txt
```

**Requirements:**
- Python 3.9+
- 16GB RAM recommended
- ~10 minutes for main evaluation

---

## Quick Start

### Run Main Evaluation (99 samples)
```bash
cd experiments
python swe_bench_mirror_evaluator.py
# Output: 68.7% accuracy, 76.1% TPR, 50.0% FPR
```

### Run Ablation Study (15 configurations)
```bash
python ablation_study.py
# Output: +39.7pp multi-agent advantage
```

### Generate Claude Patches (requires API key)
```bash
# Add ANTHROPIC_API_KEY to .env
python generate_claude_patches.py
# Generates 300 patches (~260 min, ~$25)
```

---

## Project Structure

```
codex-verify/
├── src/                  # Core implementation (6,122 lines)
│   ├── agents/          # 4 verification agents
│   └── orchestration/   # Async orchestration, aggregation
├── experiments/         # Evaluation scripts
│   ├── swe_bench_mirror_evaluator.py  # Main eval
│   ├── ablation_study.py               # Ablation study
│   └── generate_claude_patches.py      # Patch generation
├── results/             # Experimental outputs
│   ├── ablation/       # Ablation results
│   ├── claude_patches/ # 300 Claude patch evaluations
│   └── mirror_eval/    # 99-sample benchmark results
├── paper/               # LaTeX paper source
├── tests/               # Unit & integration tests
├── ui/                  # Streamlit dashboard
└── docs/                # Documentation
```

---

## Results Summary

### Main Evaluation (99 samples, perfect ground truth)
| Metric | Value | Comparison |
|--------|-------|------------|
| Accuracy | 68.7% ± 9.1% | +28.7pp over Codex (40%) |
| TPR (bug detection) | 76.1% | Matches Meta Prompt (75%) |
| FPR (false alarms) | 50.0% | Higher than Meta (8.6%), static vs dynamic tradeoff |
| F1 Score | 0.777 | Balanced precision-recall |

### Ablation Study (15 configurations)
| Agent Count | Avg Accuracy | Marginal Gain |
|-------------|--------------|---------------|
| 1 agent | 32.8% | baseline |
| 2 agents | 47.7% | +14.9pp |
| 3 agents | 61.2% | +13.5pp |
| 4 agents | 72.4% | +11.2pp |

**Total multi-agent advantage: +39.7pp**

### Real-World Validation (300 Claude Sonnet 4.5 patches)
- 72% flagged as FAIL
- 23% flagged as WARNING
- 2% passed verification
- Demonstrates strict enterprise standards

---

## Paper

**Title:** CodeX-Verify: Multi-Agent Verification of LLM-Generated Code via Compound Vulnerability Detection and Information-Theoretic Ensemble

**Authors:** Shreshth Rajan (Noumenon Labs, Harvard University)

**Status:** Submitted to ArXiv (cs.SE, cs.LG)

**Paper source:** `paper/main.tex`

---

## Citation

If you use CodeX-Verify in your research, please cite:

```bibtex
@article{rajan2025codexverify,
  title={CodeX-Verify: Multi-Agent Verification of LLM-Generated Code via Compound Vulnerability Detection and Information-Theoretic Ensemble},
  author={Rajan, Shreshth},
  journal={arXiv preprint},
  year={2025}
}
```

---

## License

MIT License - see LICENSE file

---

## Contact

**Shreshth Rajan**
- Email: shreshthrajan@college.harvard.edu
- GitHub: [@ShreshthRajan](https://github.com/ShreshthRajan)
- Noumenon Labs, Harvard University

---

## Acknowledgments

Built on research from SWE-bench, Meta Prompt Testing, and multi-agent systems literature. See paper for complete references.
