"""
Claude Patch Generation & Evaluation System

Generates patches using Claude Sonnet 4.5 for SWE-bench issues, then evaluates with CodeX-Verify.
This provides REAL LLM-generated data for publication-quality evaluation.

Methodology:
- Load 300 SWE-bench Lite issues
- Generate patch with Claude for each issue
- Evaluate patch with our calibrated verification system
- Compare results, calculate publication metrics

Security: Reads API key from .env file (gitignored)
"""

import asyncio
import time
import json
import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.orchestration.async_orchestrator import AsyncOrchestrator, VerificationConfig

# Import Anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("❌ anthropic library required: pip install anthropic")
    sys.exit(1)

# Import datasets
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("❌ datasets library required: pip install datasets")
    sys.exit(1)


@dataclass
class ClaudePatchSample:
    """A single sample with Claude-generated patch"""
    instance_id: str
    repo: str
    problem_statement: str
    hints_text: Optional[str]

    # Claude generation
    claude_patch: str
    generation_time: float
    model_used: str

    # Verification results
    verification_score: float
    verification_verdict: str
    issues_found: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int

    # Agent breakdown
    agent_scores: Dict[str, float]

    # Detailed analysis
    key_issues: List[str]
    verification_time: float

    # Quality assessment
    patch_quality_estimate: str  # HIGH, MEDIUM, LOW based on our analysis


class ClaudePatchGenerator:
    """
    Generates patches using Claude Sonnet 4.5 and evaluates them.

    Designed for publication-quality evaluation with proper rate limiting,
    error handling, and progress tracking.
    """

    def __init__(self, sample_size: int = 300, model: str = "claude-sonnet-4-5-20250929"):
        self.sample_size = sample_size
        self.model = model

        # Initialize Anthropic client
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment. Set it in .env file.")

        self.client = anthropic.Anthropic(api_key=api_key)

        # Initialize verifier with calibrated settings
        self.config = VerificationConfig.default()
        self.orchestrator = AsyncOrchestrator(self.config)

        # Rate limiting (respectful API usage)
        self.rate_limit_delay = 2.0  # 2 seconds between requests (conservative)
        self.max_retries = 3

        # Progress tracking
        self.generated_count = 0
        self.evaluated_count = 0
        self.failed_count = 0
        self.start_time = None

        # Results storage
        self.results: List[ClaudePatchSample] = []

    async def generate_patch(self, problem_statement: str, repo: str,
                            hints: Optional[str] = None) -> str:
        """
        Generate a patch using Claude Sonnet 4.5.

        Prompt engineering based on SWE-bench best practices.
        """

        # Construct prompt
        prompt_parts = [
            "You are an expert software engineer fixing a bug in a production codebase.",
            "",
            f"Repository: {repo}",
            "",
            "Issue Description:",
            problem_statement,
        ]

        if hints:
            prompt_parts.extend([
                "",
                "Additional Context:",
                hints
            ])

        prompt_parts.extend([
            "",
            "Generate a Python code patch that fixes this issue.",
            "Requirements:",
            "- Generate ONLY valid Python code (no markdown, no explanations)",
            "- Make it production-ready with proper error handling",
            "- Include necessary imports",
            "- Handle edge cases appropriately",
            "",
            "Return just the code patch:"
        ])

        prompt = "\n".join(prompt_parts)

        # Call Claude API with retries
        for attempt in range(self.max_retries):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    temperature=0.3,  # Lower temperature for more reliable code
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )

                # Extract code from response
                patch = message.content[0].text

                # Clean up any markdown code blocks if present
                if "```python" in patch:
                    patch = patch.split("```python")[1].split("```")[0].strip()
                elif "```" in patch:
                    patch = patch.split("```")[1].split("```")[0].strip()

                return patch

            except anthropic.RateLimitError:
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 10  # Exponential backoff
                    print(f"    ⏳ Rate limit hit, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"    ⚠️  Attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(2)
                else:
                    raise

        raise Exception("Failed to generate patch after all retries")

    async def evaluate_patch(self, patch: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate patch with our calibrated verification system"""

        start_time = time.time()

        try:
            report = await self.orchestrator.verify_code(patch, context)

            # Extract metrics
            critical_count = len([i for i in report.aggregated_issues if i.severity.value == 'critical'])
            high_count = len([i for i in report.aggregated_issues if i.severity.value == 'high'])
            medium_count = len([i for i in report.aggregated_issues if i.severity.value == 'medium'])
            low_count = len([i for i in report.aggregated_issues if i.severity.value == 'low'])

            # Extract top 3 critical/high issues for analysis
            key_issues = []
            for issue in report.aggregated_issues:
                if issue.severity.value in ['critical', 'high'] and len(key_issues) < 3:
                    key_issues.append(f"{issue.severity.value.upper()}: {issue.type} - {issue.message[:100]}")

            verification_time = time.time() - start_time

            # Estimate patch quality based on our analysis
            if critical_count > 0:
                quality_estimate = "LOW"
            elif high_count >= 3:
                quality_estimate = "LOW"
            elif high_count >= 1 or medium_count >= 5:
                quality_estimate = "MEDIUM"
            else:
                quality_estimate = "HIGH"

            return {
                'verification_score': report.overall_score,
                'verification_verdict': report.overall_status,
                'issues_found': len(report.aggregated_issues),
                'critical_issues': critical_count,
                'high_issues': high_count,
                'medium_issues': medium_count,
                'low_issues': low_count,
                'agent_scores': {name: result.overall_score for name, result in report.agent_results.items()},
                'key_issues': key_issues,
                'verification_time': verification_time,
                'quality_estimate': quality_estimate
            }

        except Exception as e:
            return {
                'verification_score': 0.0,
                'verification_verdict': 'ERROR',
                'issues_found': 0,
                'critical_issues': 0,
                'high_issues': 0,
                'medium_issues': 0,
                'low_issues': 0,
                'agent_scores': {},
                'key_issues': [f"Verification error: {str(e)}"],
                'verification_time': time.time() - start_time,
                'quality_estimate': 'ERROR'
            }

    async def process_single_issue(self, issue_data: Dict[str, Any], index: int, total: int) -> Optional[ClaudePatchSample]:
        """Process a single SWE-bench issue: generate patch + evaluate"""

        instance_id = issue_data.get('instance_id', f'unknown_{index}')
        repo = issue_data.get('repo', 'unknown')
        problem = issue_data.get('problem_statement', '')
        hints = issue_data.get('hints_text')

        print(f"\n🔍 [{index:3d}/{total}] {instance_id}")
        print(f"    📁 {repo}")
        print(f"    📝 {problem[:80]}{'...' if len(problem) > 80 else ''}")

        try:
            # Generate patch with Claude
            print(f"    🤖 Generating patch with Claude...")
            gen_start = time.time()
            claude_patch = await self.generate_patch(problem, repo, hints)
            gen_time = time.time() - gen_start

            self.generated_count += 1
            print(f"    ✅ Generated ({gen_time:.1f}s) - {len(claude_patch)} chars")

            # Evaluate with our system
            print(f"    🔬 Verifying patch...")
            context = {
                'instance_id': instance_id,
                'repo': repo,
                'problem_statement': problem,
                'claude_generated': True,
                'model': self.model,
                'swe_bench_sample': True,  # CRITICAL: Triggers patch_context mode for appropriate thresholds
                'github_issue': True  # Additional signal for patch evaluation
            }

            eval_results = await self.evaluate_patch(claude_patch, context)
            self.evaluated_count += 1

            # Display results
            verdict_icon = "✅" if eval_results['verification_verdict'] == "PASS" else "⚠️" if eval_results['verification_verdict'] == "WARNING" else "❌"
            print(f"    {verdict_icon} Verdict: {eval_results['verification_verdict']} | Score: {eval_results['verification_score']:.3f}")
            print(f"    📊 Issues: {eval_results['issues_found']} ({eval_results['critical_issues']}C, {eval_results['high_issues']}H)")
            print(f"    🎯 Quality Estimate: {eval_results['quality_estimate']}")

            # Create sample record
            sample = ClaudePatchSample(
                instance_id=instance_id,
                repo=repo,
                problem_statement=problem[:500],  # Truncate for storage
                hints_text=hints[:200] if hints else None,
                claude_patch=claude_patch,
                generation_time=gen_time,
                model_used=self.model,
                verification_score=eval_results['verification_score'],
                verification_verdict=eval_results['verification_verdict'],
                issues_found=eval_results['issues_found'],
                critical_issues=eval_results['critical_issues'],
                high_issues=eval_results['high_issues'],
                medium_issues=eval_results['medium_issues'],
                low_issues=eval_results['low_issues'],
                agent_scores=eval_results['agent_scores'],
                key_issues=eval_results['key_issues'],
                verification_time=eval_results['verification_time'],
                patch_quality_estimate=eval_results['quality_estimate']
            )

            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)

            return sample

        except Exception as e:
            self.failed_count += 1
            print(f"    ❌ ERROR: {str(e)}")
            return None

    async def run_generation_and_evaluation(self) -> Dict[str, Any]:
        """Main execution: generate 300 patches and evaluate all"""

        self.start_time = time.time()

        print("🚀 CLAUDE PATCH GENERATION & EVALUATION")
        print("Generating real LLM patches for publication-quality evaluation")
        print("=" * 90)
        print()

        # Check API key
        if not os.getenv('ANTHROPIC_API_KEY'):
            print("❌ ANTHROPIC_API_KEY not set in environment")
            print("   Create .env file with: ANTHROPIC_API_KEY=your-key")
            return {'error': 'API key not configured'}

        print("✅ Anthropic API configured")
        print(f"🤖 Model: {self.model}")
        print(f"📊 Target: {self.sample_size} patches")
        print(f"⏱️  Estimated time: {self.sample_size * self.rate_limit_delay / 60:.0f} minutes")
        print()

        # Load SWE-bench Lite
        print("📥 Loading SWE-bench Lite dataset...")
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test", streaming=True)
        issues = list(dataset.take(self.sample_size))

        print(f"✅ Loaded {len(issues)} issues")
        print()
        print("🎯 Starting generation and evaluation...")
        print("   This will take ~{:.0f} minutes with rate limiting".format(len(issues) * self.rate_limit_delay / 60))
        print()

        # Process each issue
        for i, issue_data in enumerate(issues, 1):
            sample = await self.process_single_issue(issue_data, i, len(issues))

            if sample:
                self.results.append(sample)

            # Progress update every 10 samples
            if i % 10 == 0:
                elapsed = time.time() - self.start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (len(issues) - i) / rate if rate > 0 else 0

                print()
                print(f"📈 Progress: {i}/{len(issues)} ({i/len(issues)*100:.1f}%)")
                print(f"   ✅ Generated: {self.generated_count} | 🔬 Evaluated: {self.evaluated_count} | ❌ Failed: {self.failed_count}")
                print(f"   ⏱️  Elapsed: {elapsed/60:.1f}min | Remaining: ~{remaining/60:.1f}min")

                # Save intermediate results (checkpoint)
                if i % 50 == 0:
                    await self._save_checkpoint(i)

        # Calculate final statistics
        total_time = time.time() - self.start_time
        stats = self._calculate_statistics()

        # Save final results
        await self._save_final_results(stats, total_time)

        # Display summary
        self._display_summary(stats, total_time)

        return {
            'total_samples': len(self.results),
            'statistics': stats,
            'execution_time': total_time
        }

    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive statistics for publication"""

        if not self.results:
            return {'error': 'no results'}

        # Basic stats
        total = len(self.results)

        # Verdict distribution
        verdict_counts = {}
        for sample in self.results:
            verdict = sample.verification_verdict
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        # Quality estimate distribution
        quality_counts = {}
        for sample in self.results:
            quality = sample.patch_quality_estimate
            quality_counts[quality] = quality_counts.get(quality, 0) + 1

        # Score statistics
        scores = [s.verification_score for s in self.results]

        # Issue statistics
        total_issues = sum(s.issues_found for s in self.results)
        critical_issues = sum(s.critical_issues for s in self.results)
        high_issues = sum(s.high_issues for s in self.results)

        # Agent performance
        agent_stats = {}
        for sample in self.results:
            for agent_name, score in sample.agent_scores.items():
                if agent_name not in agent_stats:
                    agent_stats[agent_name] = []
                agent_stats[agent_name].append(score)

        agent_performance = {}
        for agent_name, agent_scores in agent_stats.items():
            agent_performance[agent_name] = {
                'mean': statistics.mean(agent_scores),
                'median': statistics.median(agent_scores),
                'std': statistics.stdev(agent_scores) if len(agent_scores) > 1 else 0,
                'min': min(agent_scores),
                'max': max(agent_scores)
            }

        # Repository breakdown
        repo_stats = {}
        for sample in self.results:
            repo = sample.repo
            if repo not in repo_stats:
                repo_stats[repo] = {'count': 0, 'avg_score': 0, 'pass_rate': 0}
            repo_stats[repo]['count'] += 1

        for repo in repo_stats:
            repo_samples = [s for s in self.results if s.repo == repo]
            repo_stats[repo]['avg_score'] = statistics.mean([s.verification_score for s in repo_samples])
            repo_stats[repo]['pass_rate'] = len([s for s in repo_samples if s.verification_verdict == 'PASS']) / len(repo_samples)

        # Time statistics
        gen_times = [s.generation_time for s in self.results]
        verify_times = [s.verification_time for s in self.results]

        return {
            'summary': {
                'total_samples': total,
                'generated': self.generated_count,
                'evaluated': self.evaluated_count,
                'failed': self.failed_count,
                'success_rate': self.evaluated_count / total if total > 0 else 0
            },
            'verdict_distribution': verdict_counts,
            'quality_distribution': quality_counts,
            'score_statistics': {
                'mean': statistics.mean(scores),
                'median': statistics.median(scores),
                'std': statistics.stdev(scores) if len(scores) > 1 else 0,
                'min': min(scores),
                'max': max(scores),
                'q25': sorted(scores)[len(scores)//4] if scores else 0,
                'q75': sorted(scores)[3*len(scores)//4] if scores else 0
            },
            'issue_statistics': {
                'total_issues': total_issues,
                'critical_issues': critical_issues,
                'high_issues': high_issues,
                'avg_issues_per_sample': total_issues / total if total > 0 else 0,
                'samples_with_critical': len([s for s in self.results if s.critical_issues > 0]),
                'samples_with_high': len([s for s in self.results if s.high_issues > 0])
            },
            'agent_performance': agent_performance,
            'repository_breakdown': dict(sorted(repo_stats.items(), key=lambda x: x[1]['count'], reverse=True)),
            'timing_statistics': {
                'avg_generation_time': statistics.mean(gen_times),
                'avg_verification_time': statistics.mean(verify_times),
                'total_generation_time': sum(gen_times),
                'total_verification_time': sum(verify_times)
            }
        }

    async def _save_checkpoint(self, processed_count: int):
        """Save intermediate checkpoint"""
        checkpoint_file = f"claude_patches_checkpoint_{processed_count}.json"

        checkpoint_data = {
            'checkpoint_at': processed_count,
            'timestamp': datetime.now().isoformat(),
            'results_so_far': [asdict(r) for r in self.results]
        }

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        print(f"    💾 Checkpoint saved: {checkpoint_file}")

    async def _save_final_results(self, stats: Dict[str, Any], total_time: float):
        """Save comprehensive final results"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Detailed results file
        results_file = f"claude_patch_results_{timestamp}.json"
        detailed_data = {
            'metadata': {
                'timestamp': timestamp,
                'evaluation_type': 'claude_generated_patches',
                'model': self.model,
                'total_samples': len(self.results),
                'sample_size_target': self.sample_size,
                'total_execution_time': total_time,
                'methodology': 'Generate patches with Claude Sonnet 4.5, evaluate with CodeX-Verify'
            },
            'statistics': stats,
            'detailed_results': [asdict(r) for r in self.results]
        }

        with open(results_file, 'w') as f:
            json.dump(detailed_data, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {results_file}")

        # Summary report
        summary_file = f"claude_patch_summary_{timestamp}.md"
        await self._generate_summary_report(stats, summary_file, total_time)
        print(f"📋 Summary saved to: {summary_file}")

    async def _generate_summary_report(self, stats: Dict[str, Any], filename: str, total_time: float):
        """Generate markdown summary report"""

        report = f"""# Claude Patch Generation & Evaluation Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

Generated and evaluated **{stats['summary']['total_samples']} patches** using Claude Sonnet 4.5 for real SWE-bench issues.

### Key Metrics
- **Success Rate**: {stats['summary']['success_rate']:.1%}
- **Average Verification Score**: {stats['score_statistics']['mean']:.3f}
- **Total Issues Detected**: {stats['issue_statistics']['total_issues']}
- **Critical Issues**: {stats['issue_statistics']['critical_issues']}

## Verdict Distribution
"""
        for verdict, count in stats['verdict_distribution'].items():
            pct = count / stats['summary']['total_samples'] * 100
            report += f"- **{verdict}**: {count} samples ({pct:.1f}%)\n"

        report += f"""
## Quality Distribution (Our Estimate)
"""
        for quality, count in stats['quality_distribution'].items():
            pct = count / stats['summary']['total_samples'] * 100
            report += f"- **{quality}**: {count} samples ({pct:.1f}%)\n"

        report += f"""
## Agent Performance
"""
        for agent, perf in stats['agent_performance'].items():
            report += f"""### {agent.title()}
- Mean Score: {perf['mean']:.3f}
- Range: {perf['min']:.3f} - {perf['max']:.3f}
- Std Dev: {perf['std']:.3f}

"""

        report += f"""
## Top Repositories
"""
        for repo, repo_stats in list(stats['repository_breakdown'].items())[:10]:
            report += f"- **{repo}**: {repo_stats['count']} samples (avg score: {repo_stats['avg_score']:.3f}, pass rate: {repo_stats['pass_rate']:.1%})\n"

        report += f"""
## Performance
- **Total Execution Time**: {total_time:.1f} seconds ({total_time/60:.1f} minutes)
- **Avg Generation Time**: {stats['timing_statistics']['avg_generation_time']:.2f}s
- **Avg Verification Time**: {stats['timing_statistics']['avg_verification_time']:.2f}s

## Conclusion

Successfully generated and evaluated {stats['summary']['total_samples']} Claude patches, providing real LLM-generated data for publication-quality evaluation.
"""

        with open(filename, 'w') as f:
            f.write(report)

    def _display_summary(self, stats: Dict[str, Any], total_time: float):
        """Display final summary"""

        print("\n" + "=" * 90)
        print("🎉 CLAUDE PATCH GENERATION & EVALUATION COMPLETE!")
        print("=" * 90)
        print()
        print("📊 FINAL SUMMARY:")
        print(f"   ✅ Successfully processed: {self.evaluated_count}/{self.sample_size}")
        print(f"   ❌ Failed: {self.failed_count}")
        print(f"   ⏱️  Total time: {total_time/60:.1f} minutes")
        print()
        print(f"📈 VERIFICATION RESULTS:")
        for verdict, count in stats['verdict_distribution'].items():
            pct = count / stats['summary']['total_samples'] * 100
            icon = "✅" if verdict == "PASS" else "⚠️" if verdict == "WARNING" else "❌"
            print(f"   {icon} {verdict}: {count} ({pct:.1f}%)")
        print()
        print(f"🎯 QUALITY ESTIMATES:")
        for quality, count in stats['quality_distribution'].items():
            pct = count / stats['summary']['total_samples'] * 100
            icon = "🌟" if quality == "HIGH" else "📊" if quality == "MEDIUM" else "⚠️"
            print(f"   {icon} {quality}: {count} ({pct:.1f}%)")
        print()
        print(f"💡 AVERAGE SCORE: {stats['score_statistics']['mean']:.3f}")
        print(f"📊 SCORE RANGE: {stats['score_statistics']['min']:.3f} - {stats['score_statistics']['max']:.3f}")
        print()
        print("✅ Results ready for publication analysis!")


async def main():
    """Run Claude patch generation"""

    # Default to 300 samples (SWE-bench Lite size)
    SAMPLE_SIZE = 300

    print("🎯 CodeX-Verify: Claude Patch Generation")
    print("=" * 90)
    print()

    # Initialize generator
    try:
        generator = ClaudePatchGenerator(sample_size=SAMPLE_SIZE)
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return

    # Run generation and evaluation
    try:
        results = await generator.run_generation_and_evaluation()

        if 'error' in results:
            print(f"❌ Execution failed: {results['error']}")
            return

        print("\n🎊 SUCCESS!")
        print("📂 Results saved - ready for next phase (calibration)")

    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
        print(f"📊 Partial results: {generator.evaluated_count} samples processed")

        # Save partial results
        if generator.results:
            partial_file = f"claude_patches_partial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(partial_file, 'w') as f:
                json.dump([asdict(r) for r in generator.results], f, indent=2, default=str)
            print(f"💾 Partial results saved to: {partial_file}")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await generator.orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
