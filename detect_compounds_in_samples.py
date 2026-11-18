#!/usr/bin/env python3
"""
Directly analyze code samples for compound vulnerabilities.
Checks for specific attack chain patterns without full evaluation.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

def check_sql_injection(code: str) -> bool:
    """Check for SQL injection patterns"""
    patterns = [
        r'execute\([^)]*%',  # execute with %
        r'f["\']SELECT.*\{',  # f-string in SELECT
        r'\.format\([^)]*\).*SELECT',  # format() with SELECT
    ]
    return any(re.search(p, code, re.IGNORECASE) for p in patterns)

def check_hardcoded_credentials(code: str) -> bool:
    """Check for hardcoded credentials/secrets"""
    patterns = [
        r'password\s*=\s*["\'][^"\']{5,}["\']',
        r'api[_-]?key\s*=\s*["\'][^"\']{10,}["\']',
        r'secret\s*=\s*["\'][^"\']{10,}["\']',
        r'token\s*=\s*["\'][^"\']{10,}["\']',
        r'sk-[a-zA-Z0-9]{20,}',  # API key pattern
    ]
    return any(re.search(p, code, re.IGNORECASE) for p in patterns)

def check_code_execution(code: str) -> bool:
    """Check for eval/exec patterns"""
    patterns = [
        r'\beval\s*\(',
        r'\bexec\s*\(',
        r'__import__\s*\(',
        r'compile\s*\(',
    ]
    return any(re.search(p, code) for p in patterns)

def check_dangerous_imports(code: str) -> bool:
    """Check for dangerous imports"""
    patterns = [
        r'import\s+os\b',
        r'import\s+subprocess',
        r'import\s+sys\b',
        r'from\s+os\s+import',
    ]
    return any(re.search(p, code) for p in patterns)

def check_deserialization(code: str) -> bool:
    """Check for unsafe deserialization"""
    patterns = [
        r'pickle\.loads?\(',
        r'yaml\.load\(',
        r'marshal\.loads?\(',
    ]
    return any(re.search(p, code) for p in patterns)

def detect_compounds_in_code(code: str, problem_id: str) -> List[Tuple[str, str]]:
    """Detect compound vulnerability patterns in code"""
    compounds = []

    # Pattern 1: SQL injection + credentials
    if check_sql_injection(code) and check_hardcoded_credentials(code):
        compounds.append(('sql_injection', 'hardcoded_credentials'))

    # Pattern 2: Code execution + dangerous imports
    if check_code_execution(code) and check_dangerous_imports(code):
        compounds.append(('code_execution', 'dangerous_imports'))

    # Pattern 3: Deserialization + (execution or imports)
    if check_deserialization(code):
        if check_code_execution(code):
            compounds.append(('deserialization', 'code_execution'))
        if check_dangerous_imports(code):
            compounds.append(('deserialization', 'dangerous_imports'))

    return compounds

def main():
    """Analyze all samples for compound vulnerabilities"""

    # Load the sample data
    results_file = Path('results/combined/combined_100_samples_evaluation_20251024_114747.json')

    if not results_file.exists():
        print(f"❌ Results file not found")
        return

    with open(results_file) as f:
        data = json.load(f)

    print("=" * 80)
    print("🔍 COMPOUND VULNERABILITY DETECTION")
    print("=" * 80)
    print(f"\nAnalyzing {data['metadata']['total_samples']} code samples...")
    print()

    # We need the actual CODE for each sample
    # The results JSON doesn't store code, only verdicts
    # Need to load samples from source

    print("⚠️  Results JSON doesn't include sample code.")
    print("Need to load actual sample files to check for compounds.")
    print()

    # Check if samples directory exists
    samples_dir = Path('samples')
    if samples_dir.exists():
        sample_files = list(samples_dir.glob('**/*.py'))
        print(f"Found {len(sample_files)} sample files in samples/")
    else:
        print("No samples/ directory found.")

    print()
    print("=" * 80)
    print("TO RUN COMPOUND DETECTION PROPERLY:")
    print("=" * 80)
    print("1. Need access to actual code samples (not just results)")
    print("2. Or re-run evaluation with compound detection enabled")
    print("3. Or manually inspect the samples in your dataset")
    print()
    print("Sample categories that might have compounds:")
    print("  - security: 19 samples (check for SQL+creds, exec+import)")
    print("  - resource_management: 13 samples")
    print("  - input_validation_security: 3 samples")
    print()

if __name__ == '__main__':
    main()
