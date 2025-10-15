# CODEX-VERIFY Real SWE-bench Validation Report

Generated: 2025-10-12 18:23:45

## Executive Summary

**CODEX-VERIFY** was evaluated against **300 real GitHub issues** from the SWE-bench Lite dataset, representing actual production bugs from popular open-source repositories.

### Key Results
- **Evaluation Success Rate**: 99.7%
- **Average Score**: 0.405
- **Issue Detection Rate**: 100.0%
- **Total Issues Found**: 4496
- **Critical Issues Detected**: 294

## Performance Analysis

### Score Distribution
- **Mean Score**: 0.405
- **Median Score**: 0.400
- **Score Range**: 0.400 - 1.000

### Quality Assessment
- **High Quality Code** (≥0.8): 2 samples (0.7%)
- **Medium Quality Code** (0.5-0.8): 1 samples
- **Low Quality Code** (<0.5): 296 samples

### Verdict Distribution
- **FAIL**: 297 samples (99.3%)
- **PASS**: 2 samples (0.7%)

## Problem Category Analysis

### Correctness Issues
- **Total Samples**: 197
- **Average Score**: 0.000
- **Average Issues Found**: 0.0
- **FAIL Rate**: 99.5%

### Security Issues
- **Total Samples**: 16
- **Average Score**: 0.000
- **Average Issues Found**: 0.0
- **FAIL Rate**: 93.8%

### Resource Issues
- **Total Samples**: 4
- **Average Score**: 0.000
- **Average Issues Found**: 0.0
- **FAIL Rate**: 100.0%

### Performance Issues
- **Total Samples**: 10
- **Average Score**: 0.000
- **Average Issues Found**: 0.0
- **FAIL Rate**: 100.0%

### Edge_Case Issues
- **Total Samples**: 12
- **Average Score**: 0.000
- **Average Issues Found**: 0.0
- **FAIL Rate**: 100.0%

### Other Issues
- **Total Samples**: 44
- **Average Score**: 0.000
- **Average Issues Found**: 0.0
- **FAIL Rate**: 100.0%

### Data Issues
- **Total Samples**: 6
- **Average Score**: 0.000
- **Average Issues Found**: 0.0
- **FAIL Rate**: 100.0%

### Concurrency Issues
- **Total Samples**: 6
- **Average Score**: 0.000
- **Average Issues Found**: 0.0
- **FAIL Rate**: 100.0%

### Api Issues
- **Total Samples**: 4
- **Average Score**: 0.000
- **Average Issues Found**: 0.0
- **FAIL Rate**: 100.0%

## Repository Analysis

Top repositories by sample count:
- **django/django**: 114 samples (avg score: 0.403)
- **sympy/sympy**: 77 samples (avg score: 0.395)
- **matplotlib/matplotlib**: 23 samples (avg score: 0.400)
- **scikit-learn/scikit-learn**: 23 samples (avg score: 0.400)
- **pytest-dev/pytest**: 16 samples (avg score: 0.438)
- **sphinx-doc/sphinx**: 16 samples (avg score: 0.400)
- **astropy/astropy**: 6 samples (avg score: 0.400)
- **psf/requests**: 6 samples (avg score: 0.400)
- **pylint-dev/pylint**: 6 samples (avg score: 0.400)
- **pydata/xarray**: 5 samples (avg score: 0.400)

## Agent Performance

### Performance Agent
- **Mean Score**: 1.000
- **Score Range**: 1.000 - 1.000
- **Samples Processed**: 299

### Correctness Agent
- **Mean Score**: 0.499
- **Score Range**: 0.367 - 1.000
- **Samples Processed**: 299

### Style Agent
- **Mean Score**: 0.850
- **Score Range**: 0.000 - 1.000
- **Samples Processed**: 299

### Security Agent
- **Mean Score**: 0.981
- **Score Range**: 0.100 - 1.000
- **Samples Processed**: 299

## Notable Findings

### Top Issues Detected
- **import_organization**: 295 occurrences
- **production_readiness**: 294 occurrences
- **ast_analysis**: 293 occurrences
- **potential_bug**: 293 occurrences
- **execution**: 293 occurrences
- **function_naming**: 292 occurrences
- **logic**: 291 occurrences
- **variable_naming**: 291 occurrences
- **maintainability_index**: 289 occurrences
- **class_naming**: 289 occurrences

## Execution Performance
- **Total Execution Time**: 10.56 seconds
- **Average Time per Sample**: 0.02 seconds

## Conclusion

CODEX-VERIFY successfully analyzed real-world production code from GitHub repositories, demonstrating practical applicability for enterprise code verification scenarios.
