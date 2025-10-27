# Experiments

This directory contains evaluation scripts for CodeX-Verify.

## Main Evaluation Scripts

### `swe_bench_mirror_evaluator.py`
- **Purpose:** Main evaluation on 99 curated samples with perfect ground truth
- **Output:** Results in `../results/mirror_eval/`
- **Runtime:** ~10 minutes
- **Usage:** `python swe_bench_mirror_evaluator.py`
- **Key Results:** 68.7% accuracy, 76.1% TPR, 50.0% FPR

### `ablation_study.py`
- **Purpose:** Comprehensive ablation across 15 agent configurations
- **Output:** Results in `../results/ablation/`
- **Runtime:** ~2.5 hours
- **Usage:** `python ablation_study.py`
- **Key Results:** +39.7pp multi-agent advantage

### `generate_claude_patches.py`
- **Purpose:** Generate patches using Claude Sonnet 4.5 for SWE-bench issues
- **Requirements:** ANTHROPIC_API_KEY in .env file
- **Output:** Results in `../results/claude_patches/`
- **Runtime:** ~260 minutes (300 patches)
- **Cost:** ~$25 in API fees
- **Usage:** `python generate_claude_patches.py`

### `swe_bench_real_evaluator.py`
- **Purpose:** Evaluate on real SWE-bench dataset (large scale)
- **Output:** Results in `../results/`
- **Runtime:** Variable (depends on sample size)
- **Usage:** Edit SAMPLE_SIZE in file, then run

## Running Evaluations

1. Ensure dependencies installed: `pip install -r ../requirements.txt`
2. Navigate to this directory: `cd experiments`
3. Run desired evaluation script
4. Results saved to `../results/` automatically

## Reproducibility

All scripts use `random_state=42` for deterministic results. See paper Section 5.6 for complete reproduction instructions.
