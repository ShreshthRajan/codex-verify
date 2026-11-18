#!/usr/bin/env python3
"""
Quick script to analyze compound vulnerabilities in the existing 99-sample results.
No need to re-run full evaluation - just analyze what was already detected.
"""

import json
import sys
from pathlib import Path

def analyze_compounds():
    """Analyze existing results for compound vulnerabilities"""

    results_file = Path('results/combined/combined_100_samples_evaluation_20251024_114747.json')

    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return

    with open(results_file) as f:
        data = json.load(f)

    print("=" * 80)
    print("COMPOUND VULNERABILITY ANALYSIS")
    print("=" * 80)
    print(f"\nDataset: {data['metadata']['total_samples']} samples")
    print(f"  - Mirror (hand-curated): {data['metadata']['mirror_samples']}")
    print(f"  - Claude-generated: {data['metadata']['claude_samples']}")

    # Define compound patterns to look for
    compound_patterns = {
        'sql_injection + credentials': (['sql_injection', 'hardcoded_secret', 'hardcoded_password'],
                                        'SQL + Creds'),
        'code_execution + imports': (['eval_execution', 'exec_execution', 'dangerous_import'],
                                    'Exec + Import'),
        'deserialization + network': (['pickle_loads', 'yaml_load'],
                                     'Deserial + Network'),
    }

    # Analyze each sample for compound patterns
    compounds_found = []

    for sample in data['detailed_results']:
        sample_id = sample['problem_id']
        category = sample.get('category', '')
        critical = sample.get('critical_issues', 0)
        high = sample.get('high_issues', 0)

        # Check if this sample might have compounds based on category
        # (Real check would need to look at actual vulnerability types detected)

        # For now, samples with multiple high/critical issues are candidates
        if critical + high >= 3:
            compounds_found.append({
                'id': sample_id,
                'category': category,
                'critical': critical,
                'high': high,
                'total_severity': critical + high
            })

    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"Samples with 3+ high/critical issues (compound candidates): {len(compounds_found)}")

    if compounds_found:
        print("\nTop candidates:")
        for c in sorted(compounds_found, key=lambda x: x['total_severity'], reverse=True)[:10]:
            print(f"  {c['id']}: {c['critical']} critical, {c['high']} high (category: {c['category']})")

    # Check category distribution
    print(f"\n📈 CATEGORY BREAKDOWN:")
    categories = {}
    for sample in data['detailed_results']:
        cat = sample.get('category', 'unknown')
        if cat not in categories:
            categories[cat] = {'total': 0, 'buggy': 0}
        categories[cat]['total'] += 1
        if sample['should_reject']:
            categories[cat]['buggy'] += 1

    for cat, counts in sorted(categories.items(), key=lambda x: x[1]['buggy'], reverse=True)[:15]:
        print(f"  {cat}: {counts['buggy']}/{counts['total']} buggy")

    print("\n" + "=" * 80)
    print("⚠️  NOTE: This is a heuristic analysis.")
    print("Real compound detection requires checking actual vulnerability types,")
    print("not just issue counts. Need to enhance result JSON to include detected")
    print("vulnerability types for each sample.")
    print("=" * 80)

if __name__ == '__main__':
    analyze_compounds()
