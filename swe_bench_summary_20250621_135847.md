# CODEX-VERIFY Real SWE-bench Validation Report

Generated: 2025-06-21 13:58:47

## Executive Summary

**CODEX-VERIFY** was evaluated against **50 real GitHub issues** from the SWE-bench Lite dataset, representing actual production bugs from popular open-source repositories.

### Key Results
- **Evaluation Success Rate**: 100.0%
- **Average Score**: 0.411
- **Issue Detection Rate**: 100.0%
- **Total Issues Found**: 809
- **Critical Issues Detected**: 47

## Performance Analysis

### Score Distribution
- **Mean Score**: 0.411
- **Median Score**: 0.400
- **Score Range**: 0.400 - 0.967

### Quality Assessment
- **High Quality Code** (≥0.8): 1 samples (2.0%)
- **Medium Quality Code** (0.5-0.8): 0 samples
- **Low Quality Code** (<0.5): 49 samples

### Verdict Distribution
- **FAIL**: 49 samples (98.0%)
- **PASS**: 1 samples (2.0%)

## Problem Category Analysis

### Correctness Issues
- **Total Samples**: 30
- **Average Score**: 0.000
- **Average Issues Found**: 0.0
- **FAIL Rate**: 100.0%

### Security Issues
- **Total Samples**: 6
- **Average Score**: 0.000
- **Average Issues Found**: 0.0
- **FAIL Rate**: 83.3%

### Resource Issues
- **Total Samples**: 3
- **Average Score**: 0.000
- **Average Issues Found**: 0.0
- **FAIL Rate**: 100.0%

### Performance Issues
- **Total Samples**: 2
- **Average Score**: 0.000
- **Average Issues Found**: 0.0
- **FAIL Rate**: 100.0%

### Edge_Case Issues
- **Total Samples**: 3
- **Average Score**: 0.000
- **Average Issues Found**: 0.0
- **FAIL Rate**: 100.0%

### Other Issues
- **Total Samples**: 5
- **Average Score**: 0.000
- **Average Issues Found**: 0.0
- **FAIL Rate**: 100.0%

### Data Issues
- **Total Samples**: 1
- **Average Score**: 0.000
- **Average Issues Found**: 0.0
- **FAIL Rate**: 100.0%

## Repository Analysis

Top repositories by sample count:
- **django/django**: 44 samples (avg score: 0.413)
- **astropy/astropy**: 6 samples (avg score: 0.400)

## Agent Performance

### Performance Agent
- **Mean Score**: 1.000
- **Score Range**: 1.000 - 1.000
- **Samples Processed**: 50

### Style Agent
- **Mean Score**: 0.667
- **Score Range**: 0.000 - 1.000
- **Samples Processed**: 50

### Correctness Agent
- **Mean Score**: 0.502
- **Score Range**: 0.367 - 1.000
- **Samples Processed**: 50

### Security Agent
- **Mean Score**: 0.976
- **Score Range**: 0.800 - 1.000
- **Samples Processed**: 50

## Notable Findings

### Top Issues Detected
- **flake8**: 59 occurrences
- **production_readiness**: 49 occurrences
- **execution**: 49 occurrences
- **import_organization**: 49 occurrences
- **logic**: 48 occurrences
- **potential_bug**: 48 occurrences
- **variable_naming**: 48 occurrences
- **function_naming**: 48 occurrences
- **ast_analysis**: 47 occurrences
- **maintainability_index**: 47 occurrences

## Execution Performance
- **Total Execution Time**: 5.21 seconds
- **Average Time per Sample**: 0.07 seconds

## Conclusion

CODEX-VERIFY successfully analyzed real-world production code from GitHub repositories, demonstrating practical applicability for enterprise code verification scenarios.
