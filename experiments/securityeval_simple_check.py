#!/usr/bin/env python3
"""
Quick SecurityEval Analysis - No System Imports Needed

Just checks how many SecurityEval samples YOUR security patterns would catch.
Uses the same patterns from your Security agent.
"""

import json
import re
from pathlib import Path
from typing import Dict, List

# Your Security agent's patterns (from src/agents/security_auditor.py)
SECURITY_PATTERNS = {
    'sql_injection': [
        r'execute\([^)]*%',
        r'\.format\([^)]*\).*SELECT',
        r'f["\'].*SELECT.*\{',
    ],
    'command_injection': [
        r'os\.system\s*\(',
        r'subprocess.*shell\s*=\s*True',
    ],
    'code_execution': [
        r'\beval\s*\(',
        r'\bexec\s*\(',
    ],
    'unsafe_deserialization': [
        r'pickle\.loads?\s*\(',
        r'yaml\.load\s*\(',
    ],
    'weak_crypto': [
        r'hashlib\.md5\s*\(',
        r'hashlib\.sha1\s*\(',
        r'random\.randint',
    ],
    'hardcoded_secrets': [
        r'password\s*=\s*["\'][^"\']{5,}["\']',
        r'api[_-]?key\s*=\s*["\'][^"\']{10,}["\']',
    ],
}

def check_patterns(code: str) -> Dict:
    """Check if code matches any security patterns"""
    matches = []

    for category, patterns in SECURITY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                matches.append(category)
                break  # One match per category

    return {
        'detected': len(matches) > 0,
        'categories': matches,
        'count': len(matches)
    }

def main():
    """Analyze SecurityEval with your patterns"""

    dataset_file = Path('data/SecurityEval/dataset.jsonl')

    if not dataset_file.exists():
        print("❌ SecurityEval not found")
        print("Run: git clone https://github.com/s2e-lab/SecurityEval data/SecurityEval")
        return

    # Load SecurityEval
    samples = []
    with open(dataset_file) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"SecurityEval: {len(samples)} samples")
    print()

    # Check how many you'd catch
    detected = 0
    by_cwe = {}

    for sample in samples:
        code = sample.get('Insecure_code', '')
        cwe = sample.get('CWE', 'Unknown')

        result = check_patterns(code)

        if result['detected']:
            detected += 1

        if cwe not in by_cwe:
            by_cwe[cwe] = {'total': 0, 'detected': 0}
        by_cwe[cwe]['total'] += 1
        if result['detected']:
            by_cwe[cwe]['detected'] += 1

    tpr = detected / len(samples) if samples else 0

    print("=" * 80)
    print("RESULTS - YOUR PATTERNS ON SECURITYEVAL")
    print("=" * 80)
    print(f"Detected: {detected}/{len(samples)} ({tpr:.1%})")
    print()

    print("Coverage by CWE (top 10):")
    sorted_cwes = sorted(by_cwe.items(), key=lambda x: x[1]['detected'], reverse=True)
    for cwe, stats in sorted_cwes[:10]:
        pct = stats['detected'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {cwe}: {stats['detected']}/{stats['total']} ({pct:.0%})")

    print()
    print("=" * 80)
    print(f"ESTIMATED TPR ON SECURITYEVAL: {tpr:.1%}")
    print("=" * 80)
    print()
    print("This tests your Security agent only (not Correctness/Performance/Style)")
    print("SecurityEval is security-focused, so other agents don't apply.")
    print()
    print(f"Result: Your Security patterns catch {tpr:.0%} of SecurityEval vulnerabilities")
    print("This is an honest baseline showing coverage of common CWEs.")

if __name__ == '__main__':
    main()
