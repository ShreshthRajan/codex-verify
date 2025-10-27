"""
Generate Additional 101 Samples to Reach 200 Total

Current: 99 samples (29 mirror + 70 Claude)
Target: 200 samples
Need: 101 more

This gives us proper train/test split:
- Train: 100 samples (optimize thresholds)
- Test: 100 samples (report metrics)
"""

import asyncio
import json
import os
import sys
from dotenv import load_dotenv
from datetime import datetime
import time

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import anthropic
from swe_bench_mirror_evaluator import create_comprehensive_samples


async def generate_batch(client, batch_num: int, start_id: int):
    """Generate one batch of 10 samples"""

    existing = create_comprehensive_samples()

    # Rotate examples
    idx = (batch_num * 3) % len(existing)
    examples = []

    for i in range(3):
        s = existing[(idx + i) % len(existing)]
        examples.append(f"""
Problem ID: {s.problem_id}
Category: {s.failure_category}
Bug: {s.actual_issue}
Should reject: {s.should_be_rejected}

Code:
{s.codex_solution[:400]}
""")

    example_text = "\n---\n".join(examples)

    prompt = f"""Generate 10 NEW Python code test samples.

Examples:
{example_text}

Return JSON array (10 samples):
[
  {{
    "problem_id": "batch{batch_num}_sample_N",
    "issue_description": "description",
    "repo_context": "context",
    "codex_solution": "COMPLETE valid Python code (50-1500 chars)",
    "actual_issue": "bug description OR 'None - correct code'",
    "failure_category": "security|correctness|performance|edge_case|resource_management",
    "expected_test_pass": true,
    "should_be_rejected": true or false,
    "difficulty_level": "easy|medium|hard"
  }}
]

Requirements:
- 7 BAD code (bugs), 3 GOOD code
- Subtle bugs (not syntax errors)
- Valid Python
- Mix categories

Return ONLY JSON."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )

        response = message.content[0].text

        if "```json" in response:
            json_text = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_text = response.split("```")[1].split("```")[0]
        else:
            json_text = response

        samples = json.loads(json_text)

        # Validate
        valid = []
        for sample in samples:
            if 'codex_solution' in sample and len(sample['codex_solution']) >= 50:
                try:
                    compile(sample['codex_solution'], '<string>', 'exec')
                    valid.append(sample)
                except:
                    pass

        print(f" ✅ {len(valid)}/10")
        await asyncio.sleep(2)
        return valid

    except Exception as e:
        print(f" ❌ {e}")
        return []


async def main():
    """Generate 101 more samples"""

    print("🚀 GENERATING 101 SAMPLES TO REACH 200 TOTAL")
    print("=" * 80)
    print("Target: 200 samples for proper 100/100 train/test split")
    print()

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ API key not found")
        return

    client = anthropic.Anthropic(api_key=api_key)

    print("Generating in 11 batches of ~10 each...")
    print()

    all_generated = []
    start_time = time.time()

    for batch in range(1, 12):  # 11 batches
        print(f"Batch {batch}/11:", end="")
        batch_samples = await generate_batch(client, batch, len(all_generated) + 100)
        all_generated.extend(batch_samples)
        print(f" Total: {len(all_generated)}")

    elapsed = time.time() - start_time

    print()
    print(f"✅ GENERATION COMPLETE")
    print(f"   Generated: {len(all_generated)} valid samples")
    print(f"   Time: {elapsed/60:.1f} minutes")
    print()

    # Save
    output_file = f"claude_generated_101_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, 'w') as f:
        json.dump(all_generated, f, indent=2)

    print(f"💾 Saved: {output_file}")

    # Summary
    bad = len([s for s in all_generated if s.get('should_be_rejected', True)])
    good = len(all_generated) - bad

    print()
    print(f"📊 BREAKDOWN:")
    print(f"   Bad code: {bad}")
    print(f"   Good code: {good}")
    print()
    print("✅ READY FOR: Train/test split + optimization")


if __name__ == "__main__":
    asyncio.run(main())
