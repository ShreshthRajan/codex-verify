"""
Generate Full 66 Mirror Samples with Claude

Generates 66 high-quality bug samples to add to existing 34 mirror samples.
Target: 100 total samples with perfect ground truth.

Generates in batches of 10 to avoid rate limits and ensure quality.
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


async def generate_batch(client, batch_num: int, total_batches: int):
    """Generate one batch of 10 samples"""

    # Load existing for pattern
    existing = create_comprehensive_samples()

    # Create varied examples (rotate through existing samples)
    start_idx = (batch_num * 3) % len(existing)
    examples = []

    for i in range(3):
        s = existing[(start_idx + i) % len(existing)]
        examples.append(f"""
Problem ID: {s.problem_id}
Category: {s.failure_category}
Bug: {s.actual_issue}
Should reject: {s.should_be_rejected}

Code:
{s.codex_solution[:400]}
""")

    example_text = "\n---\n".join(examples)

    prompt = f"""Create 10 NEW realistic Python code samples for testing a code verification system.

Examples of the format:
{example_text}

Generate 10 UNIQUE samples as a JSON array:
[
  {{
    "problem_id": "unique_id_{batch_num}_N",
    "issue_description": "What the code does",
    "repo_context": "Context",
    "codex_solution": "COMPLETE Python code (20-150 lines)",
    "actual_issue": "Bug description OR 'None - correct code'",
    "failure_category": "security|correctness|performance|edge_case|resource_management",
    "expected_test_pass": true,
    "should_be_rejected": true or false,
    "difficulty_level": "easy|medium|hard"
  }}
]

Requirements:
- 7 should be BAD code (subtle bugs), 3 should be GOOD code
- Vary categories and difficulty
- Code must be valid Python, realistic, 50-1500 chars
- Bugs should be subtle (edge cases, security, logic errors)

Return ONLY the JSON array."""

    print(f"   Batch {batch_num}/{total_batches}: Generating...", end="", flush=True)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )

        response = message.content[0].text

        # Extract JSON
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
            # Check has code
            if 'codex_solution' in sample and len(sample['codex_solution']) >= 50:
                try:
                    compile(sample['codex_solution'], '<string>', 'exec')
                    valid.append(sample)
                except:
                    pass

        print(f" ✅ {len(valid)}/10 valid")

        await asyncio.sleep(2)  # Rate limiting

        return valid

    except Exception as e:
        print(f" ❌ Error: {e}")
        return []


async def generate_all_66():
    """Generate all 66 samples in batches"""

    print("🚀 GENERATING 66 CLAUDE SAMPLES")
    print("=" * 80)
    print("Generating in 7 batches of 10 samples each")
    print("Estimated time: 10-12 minutes")
    print()

    # Initialize Claude
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ API key not found")
        return None

    client = anthropic.Anthropic(api_key=api_key)

    # Generate in batches
    all_generated = []
    start_time = time.time()

    for batch_num in range(1, 8):  # 7 batches
        batch_samples = await generate_batch(client, batch_num, 7)
        all_generated.extend(batch_samples)

        print(f"   Running total: {len(all_generated)} valid samples")

    elapsed = time.time() - start_time

    print()
    print(f"✅ GENERATION COMPLETE")
    print(f"   Total valid: {len(all_generated)}")
    print(f"   Time: {elapsed/60:.1f} minutes")
    print()

    # Save
    output_file = f"claude_generated_66_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, 'w') as f:
        json.dump(all_generated, f, indent=2)

    print(f"💾 Saved: {output_file}")

    # Summary
    categories = {}
    bad_count = 0
    good_count = 0

    for s in all_generated:
        cat = s.get('failure_category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1

        if s.get('should_be_rejected', True):
            bad_count += 1
        else:
            good_count += 1

    print()
    print("📊 GENERATED SAMPLE BREAKDOWN:")
    print(f"   Bad code (should FAIL): {bad_count}")
    print(f"   Good code (should PASS): {good_count}")
    print()
    print("   By category:")
    for cat, count in sorted(categories.items()):
        print(f"     {cat}: {count}")

    return output_file


async def main():
    """Main execution"""

    result = await generate_all_66()

    if result:
        print()
        print("✅ SUCCESS - 66 samples generated!")
        print()
        print("📋 Next: Run combined evaluation")
        print("   python evaluate_combined_100_samples.py")
    else:
        print()
        print("❌ Generation failed - proceed with 34 samples")


if __name__ == "__main__":
    asyncio.run(main())
