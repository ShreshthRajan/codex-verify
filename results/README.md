# Experimental Results

This directory contains all experimental outputs from CodeX-Verify evaluations.

## Directory Structure

### `ablation/`
Results from comprehensive ablation study (15 agent configurations):
- `ablation_study_results_20251024_104905.json` - Complete results for all 15 configs
- `ablation_summary_20251024_104905.md` - Human-readable summary
- `ablation_results_20251024_104905.csv` - Data for plotting
- `ablation_table_*.md` - LaTeX tables

**Key Findings:** +39.7pp multi-agent advantage, diminishing returns pattern

### `claude_patches/`
Results from 300 Claude Sonnet 4.5-generated patches:
- `claude_patch_results_20251012_235129.json` - All 300 patch evaluations
- `claude_patch_summary_20251012_235129.md` - Summary statistics
- `claude_generated_*_samples*.json` - Generated code samples

**Key Findings:** 72% flagged as FAIL, 23% WARNING, 2% PASS

### `mirror_eval/`
Results from 99-sample curated benchmark:
- `enhanced_swe_bench_results.json` - Final evaluation results

**Key Findings:** 68.7% ± 9.1% accuracy, 76.1% TPR, 50.0% FPR

### `combined/`
Results from combined dataset evaluations:
- `combined_100_samples_evaluation_20251024_114747.json` - 99 samples evaluation

## Result Files Format

All JSON files follow this structure:
```json
{
  "metadata": {...},
  "metrics": {
    "accuracy": float,
    "true_positive_rate": float,
    "false_positive_rate": float,
    ...
  },
  "detailed_results": [...]
}
```

## Citation

If you use these results, please cite:
```
@article{rajan2025codexverify,
  title={CodeX-Verify: Multi-Agent Verification of LLM-Generated Code},
  author={Rajan, Shreshth},
  year={2025}
}
```
