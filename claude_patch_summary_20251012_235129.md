# Claude Patch Generation & Evaluation Report

Generated: 2025-10-12 23:51:29

## Executive Summary

Generated and evaluated **300 patches** using Claude Sonnet 4.5 for real SWE-bench issues.

### Key Metrics
- **Success Rate**: 100.0%
- **Average Verification Score**: 0.457
- **Total Issues Detected**: 4215
- **Critical Issues**: 89

## Verdict Distribution
- **FAIL**: 216 samples (72.0%)
- **WARNING**: 69 samples (23.0%)
- **PASS**: 6 samples (2.0%)
- **ERROR**: 9 samples (3.0%)

## Quality Distribution (Our Estimate)
- **LOW**: 225 samples (75.0%)
- **MEDIUM**: 69 samples (23.0%)
- **HIGH**: 6 samples (2.0%)

## Agent Performance
### Security
- Mean Score: 0.936
- Range: 0.100 - 1.000
- Std Dev: 0.173

### Correctness
- Mean Score: 0.623
- Range: 0.367 - 1.000
- Std Dev: 0.219

### Style
- Mean Score: 0.648
- Range: 0.000 - 1.000
- Std Dev: 0.268

### Performance
- Mean Score: 0.991
- Range: 0.741 - 1.000
- Std Dev: 0.027


## Top Repositories
- **django/django**: 114 samples (avg score: 0.454, pass rate: 2.6%)
- **sympy/sympy**: 77 samples (avg score: 0.452, pass rate: 1.3%)
- **matplotlib/matplotlib**: 23 samples (avg score: 0.487, pass rate: 0.0%)
- **scikit-learn/scikit-learn**: 23 samples (avg score: 0.487, pass rate: 0.0%)
- **pytest-dev/pytest**: 17 samples (avg score: 0.447, pass rate: 5.9%)
- **sphinx-doc/sphinx**: 16 samples (avg score: 0.413, pass rate: 0.0%)
- **astropy/astropy**: 6 samples (avg score: 0.467, pass rate: 0.0%)
- **psf/requests**: 6 samples (avg score: 0.467, pass rate: 0.0%)
- **pylint-dev/pylint**: 6 samples (avg score: 0.467, pass rate: 0.0%)
- **pydata/xarray**: 5 samples (avg score: 0.400, pass rate: 0.0%)

## Performance
- **Total Execution Time**: 15614.3 seconds (260.2 minutes)
- **Avg Generation Time**: 49.53s
- **Avg Verification Time**: 0.50s

## Conclusion

Successfully generated and evaluated 300 Claude patches, providing real LLM-generated data for publication-quality evaluation.
