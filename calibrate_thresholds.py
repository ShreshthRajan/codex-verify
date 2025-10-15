"""
Threshold Calibration System

This script analyzes evaluation results and provides data-driven calibration recommendations.

The goal: Maintain high TPR (>85%) while reducing FPR (<20%) by properly categorizing issues.

Key insight: Not all issues should block deployment
- CRITICAL/HIGH (security, correctness bugs) → Block deployment ✓
- MEDIUM/LOW (style, minor issues) → Warning only ✓
- INFO (suggestions, optimizations) → Informational only ✓
"""

import json
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Any
import statistics


class ThresholdCalibrator:
    """
    Analyzes evaluation results and recommends threshold calibrations.

    Based on ground truth research:
    - Target TPR: 85-95% (catch real bugs)
    - Target FPR: 10-20% (don't over-flag good code)
    - Target accuracy: 75-85% (balanced performance)
    """

    def __init__(self, results_file: str):
        """Load results from mirror evaluation"""
        with open(results_file, 'r') as f:
            self.data = json.load(f)

        self.results = self.data.get('detailed_results', [])
        self.current_tpr = self.data.get('true_positive_rate', 0)
        self.current_fpr = self.data.get('false_positive_rate', 0)
        self.current_accuracy = self.data.get('accuracy', 0)

    def analyze_false_positives(self) -> Dict[str, Any]:
        """Analyze what's causing false positives on good code"""

        # Get false positive samples
        fps = [r for r in self.results
               if not r['should_reject'] and r['our_verdict'] == 'FAIL']

        if not fps:
            return {'message': 'No false positives found!'}

        print(f"\n🔍 ANALYZING {len(fps)} FALSE POSITIVES (good code flagged as bad)\n")

        # Issue type analysis
        issue_counts = Counter()
        issue_severity_dist = defaultdict(lambda: {'critical': 0, 'high': 0, 'medium': 0, 'low': 0})

        for fp in fps:
            for issue in fp.get('detailed_issues', []):
                issue_type = issue['type']
                severity = issue['severity']

                issue_counts[issue_type] += 1
                issue_severity_dist[issue_type][severity] += 1

        # Print top FP causes
        print("📊 TOP CAUSES OF FALSE POSITIVES:")
        print("=" * 80)
        for issue_type, count in issue_counts.most_common(15):
            severity_dist = issue_severity_dist[issue_type]
            print(f"{issue_type:35} {count:3} occurrences")
            print(f"     Severity: C:{severity_dist['critical']} H:{severity_dist['high']} M:{severity_dist['medium']} L:{severity_dist['low']}")

        return {
            'total_fps': len(fps),
            'issue_type_counts': dict(issue_counts),
            'severity_distribution': dict(issue_severity_dist)
        }

    def analyze_false_negatives(self) -> Dict[str, Any]:
        """Analyze what bugs we're missing"""

        # Get false negative samples (bugs we didn't catch)
        fns = [r for r in self.results
               if r['should_reject'] and r['our_verdict'] in ['PASS', 'WARNING']]

        if not fns:
            return {'message': 'No false negatives - catching all bugs!'}

        print(f"\n🔍 ANALYZING {len(fns)} FALSE NEGATIVES (missed bugs)\n")

        print("❌ MISSED BUGS:")
        print("=" * 80)
        for fn in fns:
            print(f"Problem: {fn['problem_id']}")
            print(f"Category: {fn['failure_category']}")
            print(f"Issue: {fn['actual_issue']}")
            print(f"Our Score: {fn['our_score']:.3f} → {fn['our_verdict']}")
            print(f"Issues Found: {fn['issues_found']} ({fn['high_issues']} high)")
            print()

        return {
            'total_fns': len(fns),
            'missed_categories': [fn['failure_category'] for fn in fns]
        }

    def generate_calibration_recommendations(self) -> List[str]:
        """Generate specific, actionable calibration recommendations"""

        recommendations = []

        print("\n🎯 CALIBRATION RECOMMENDATIONS")
        print("=" * 80)

        # Analyze current state
        print(f"Current Performance:")
        print(f"  TPR: {self.current_tpr:.1%} (Target: >85%)")
        print(f"  FPR: {self.current_fpr:.1%} (Target: <20%)")
        print(f"  Accuracy: {self.current_accuracy:.1%} (Target: >75%)")
        print()

        # Get FP analysis
        fp_analysis = self.analyze_false_positives()
        fn_analysis = self.analyze_false_negatives()

        # Generate recommendations based on analysis
        if self.current_fpr > 0.50:
            recommendations.append("🚨 CRITICAL: FPR = {:.1%} - Immediate recalibration needed".format(self.current_fpr))
            recommendations.append("")
            recommendations.append("PRIMARY ACTION: Separate style from security/correctness")
            recommendations.append("")

            # Specific threshold adjustments
            style_issues = ['import_organization', 'function_naming', 'variable_naming',
                          'code_formatting', 'spacing', 'class_naming', 'code_clarity']

            for issue_type in style_issues:
                if issue_type in fp_analysis.get('issue_type_counts', {}):
                    count = fp_analysis['issue_type_counts'][issue_type]
                    severity_dist = fp_analysis.get('severity_distribution', {}).get(issue_type, {})

                    if severity_dist.get('high', 0) > 0:
                        recommendations.append(
                            f"  → Downgrade '{issue_type}' from HIGH to LOW (style, not blocker) [{count} FPs]"
                        )
                    elif severity_dist.get('medium', 0) > 0:
                        recommendations.append(
                            f"  → Downgrade '{issue_type}' from MEDIUM to INFO (style issue) [{count} FPs]"
                        )

            recommendations.append("")
            recommendations.append("THRESHOLD ADJUSTMENTS:")
            recommendations.append("  → Security/Correctness: Keep CRITICAL/HIGH blocking deployment ✓")
            recommendations.append("  → Style/Formatting: Downgrade to LOW/INFO (warning only) ✓")
            recommendations.append("  → Performance: Keep HIGH for O(n³)+, downgrade O(n²) to MEDIUM ✓")

        elif self.current_fpr > 0.30:
            recommendations.append("⚠️  FPR needs tuning - moderate adjustments")
            recommendations.append("  → Review style issue severity levels")
            recommendations.append("  → Consider context-aware thresholds")

        if self.current_tpr < 0.85:
            recommendations.append("⚠️  TPR below target - enhance detection")
            if fn_analysis.get('missed_categories'):
                categories = set(fn_analysis['missed_categories'])
                recommendations.append(f"  → Focus on categories: {', '.join(categories)}")

        # Print recommendations
        print()
        for rec in recommendations:
            print(rec)

        return recommendations

    def export_calibration_config(self, output_file: str = 'calibration_config.json'):
        """Export recommended configuration based on analysis"""

        fp_analysis = self.analyze_false_positives()

        # Build severity reclassification map
        severity_reclassification = {}

        # Style issues should be LOW or INFO
        style_issues = ['import_organization', 'function_naming', 'variable_naming',
                       'code_formatting', 'spacing', 'class_naming', 'code_clarity',
                       'consistency', 'maintainability_index']

        for issue_type in style_issues:
            if issue_type in fp_analysis.get('issue_type_counts', {}):
                severity_reclassification[issue_type] = {
                    'current': 'AUTO',
                    'recommended': 'LOW',
                    'reason': 'Style issue - should not block deployment'
                }

        # Logic/readability issues should be MEDIUM
        quality_issues = ['logic', 'potential_bug', 'production_readiness']
        for issue_type in quality_issues:
            if issue_type in fp_analysis.get('issue_type_counts', {}):
                severity_dist = fp_analysis.get('severity_distribution', {}).get(issue_type, {})
                if severity_dist.get('high', 0) > 0:
                    severity_reclassification[issue_type] = {
                        'current': 'HIGH',
                        'recommended': 'MEDIUM',
                        'reason': 'Quality issue - warning but not critical blocker'
                    }

        # Export configuration
        config = {
            'calibration_metadata': {
                'generated_from': 'mirror_evaluation',
                'current_tpr': self.current_tpr,
                'current_fpr': self.current_fpr,
                'target_tpr': 0.90,
                'target_fpr': 0.15
            },
            'severity_reclassification': severity_reclassification,
            'threshold_adjustments': {
                'style_agent': {
                    'weight_in_final_score': 0.05,  # Reduced from 0.10
                    'blocking_threshold': 0.50,  # Only block if score <0.50
                    'note': 'Style should influence score minimally'
                },
                'security_agent': {
                    'weight_in_final_score': 0.45,  # Increased from 0.40
                    'blocking_threshold': 0.70,  # Block if security <0.70
                    'note': 'Security is paramount'
                },
                'correctness_agent': {
                    'weight_in_final_score': 0.35,  # Increased from 0.30
                    'blocking_threshold': 0.60,  # Block if correctness <0.60
                    'note': 'Correctness bugs block deployment'
                },
                'performance_agent': {
                    'weight_in_final_score': 0.15,  # Reduced from 0.20
                    'blocking_threshold': 0.50,  # Only block for severe issues
                    'note': 'Performance warnings acceptable for deployment'
                }
            },
            'deployment_decision_logic': {
                'auto_pass_threshold': 0.85,
                'auto_fail_threshold': 0.50,
                'warning_range': [0.50, 0.85],
                'critical_issue_policy': 'Zero tolerance - any critical issue blocks deployment',
                'high_issue_policy': '3+ high issues block deployment',
                'style_issue_policy': 'Never blocks deployment - warning only'
            }
        }

        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n💾 Calibration config exported to: {output_file}")
        print("📋 Use this to update agent configurations")

        return config


def main():
    """Main calibration analysis"""

    # Check for results file
    import glob

    mirror_results = glob.glob('enhanced_swe_bench_results.json')

    if not mirror_results:
        print("❌ No mirror evaluation results found!")
        print("Run this first: python swe_bench_mirror_evaluator.py")
        sys.exit(1)

    results_file = mirror_results[0]
    print(f"📂 Loading results from: {results_file}\n")

    calibrator = ThresholdCalibrator(results_file)

    # Run analysis
    calibrator.generate_calibration_recommendations()

    # Export config
    config = calibrator.export_calibration_config()

    print("\n🎯 NEXT STEPS:")
    print("1. Review calibration_config.json")
    print("2. Apply severity reclassifications to agents")
    print("3. Re-run mirror evaluation to verify improvements")
    print("4. Target: TPR >90%, FPR <15%, Accuracy >80%")


if __name__ == "__main__":
    main()
