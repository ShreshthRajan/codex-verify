"""
Generate Additional Mirror Samples Using Claude

Tests: Can Claude generate high-quality bug samples matching existing 34?

Approach:
1. Extract pattern from existing 34 samples
2. Ask Claude to generate 10 test samples
3. Validate quality
4. If good, generate remaining 56 for total of 100

Uses: Existing .env with ANTHROPIC_API_KEY
"""

import asyncio
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    print("❌ anthropic required: pip install anthropic")
    sys.exit(1)

from swe_bench_mirror_evaluator import create_comprehensive_samples


async def generate_samples_with_claude(num_samples: int = 10):
    """Generate test samples using Claude"""

    print("🎯 TESTING: Claude-Generated Mirror Samples")
    print("=" * 80)
    print()

    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in .env")
        return None

    client = anthropic.Anthropic(api_key=api_key)

    # Load existing samples for pattern
    existing = create_comprehensive_samples()
    print(f"📋 Analyzing {len(existing)} existing samples for pattern...")
    print()

    # Create example pattern from first 3 samples
    examples = []
    for s in existing[:3]:
        examples.append(f"""
Problem ID: {s.problem_id}
Category: {s.failure_category}
Difficulty: {s.difficulty_level}
Description: {s.issue_description}
Bug: {s.actual_issue}
Should reject: {s.should_be_rejected}

Code:
{s.codex_solution[:300]}...
""")

    example_text = "\n---\n".join(examples)

    # Prompt
    prompt = f"""You are creating test cases for a code verification system.

Here are 3 examples of the format:

{example_text}

Generate {num_samples} NEW code samples following this EXACT pattern:
- Each should be realistic Python code (20-100 lines)
- Include subtle bugs (edge cases, security issues, performance problems, or correctness bugs)
- Some should be GOOD code (should_reject=False), some BAD (should_reject=True)
- Vary difficulty: easy, medium, hard
- Categories: security, correctness, performance, edge_case, resource_management

Return as a JSON array with this structure:
[
  {{
    "problem_id": "unique_id",
    "issue_description": "description",
    "repo_context": "context",
    "codex_solution": "full Python code here",
    "actual_issue": "description of bug or 'None' if good code",
    "failure_category": "category",
    "difficulty_level": "easy|medium|hard",
    "should_be_rejected": true or false
  }}
]

IMPORTANT:
- problem_id must be unique (use pattern: category_number like "sql_injection_005")
- codex_solution must be complete, valid Python code
- Mix of bad code (70%) and good code (30%)
- Make bugs SUBTLE (not obvious syntax errors)

Return ONLY the JSON array, no other text."""

    print(f"🤖 Asking Claude to generate {num_samples} samples...")
    print("   (This may take 30-60 seconds)")
    print()

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8000,
            temperature=0.7,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        response = message.content[0].text

        # Extract JSON
        if "```json" in response:
            json_text = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_text = response.split("```")[1].split("```")[0]
        else:
            json_text = response

        # Parse
        generated_samples = json.loads(json_text)

        print(f"✅ Claude generated {len(generated_samples)} samples")
        print()

        # Validate quality
        print("🔬 QUALITY VALIDATION:")
        print("-" * 80)

        valid_count = 0
        issues = []

        for i, sample in enumerate(generated_samples, 1):
            # Check required fields
            required = ['problem_id', 'codex_solution', 'actual_issue',
                       'failure_category', 'should_be_rejected']

            missing = [f for f in required if f not in sample]

            if missing:
                issues.append(f"Sample {i}: Missing fields {missing}")
                continue

            # Check code is valid Python
            code = sample['codex_solution']

            try:
                compile(code, '<string>', 'exec')
                valid_syntax = True
            except:
                valid_syntax = False
                issues.append(f"Sample {i}: Invalid Python syntax")

            if len(code) < 50:
                issues.append(f"Sample {i}: Code too short ({len(code)} chars)")
            elif len(code) > 2000:
                issues.append(f"Sample {i}: Code too long ({len(code)} chars)")

            if valid_syntax and 50 <= len(code) <= 2000:
                valid_count += 1
                print(f"   ✅ Sample {i}: {sample['problem_id']} ({len(code)} chars, {sample['failure_category']})")

        print()
        print(f"📊 VALIDATION RESULTS:")
        print(f"   Valid: {valid_count}/{len(generated_samples)} ({valid_count/len(generated_samples)*100:.0f}%)")

        if issues:
            print(f"   Issues found: {len(issues)}")
            for issue in issues[:5]:
                print(f"     - {issue}")

        # Save generated samples for review
        output_file = "claude_generated_samples_test.json"
        with open(output_file, 'w') as f:
            json.dump(generated_samples, f, indent=2)

        print()
        print(f"💾 Saved to: {output_file}")
        print()

        # Decision
        if valid_count >= num_samples * 0.7:  # 70% valid
            print("✅ QUALITY ACCEPTABLE (≥70% valid)")
            print(f"   → Can generate full 66 samples")
            print(f"   → Estimated: {66 * (num_samples / valid_count):.0f} attempts needed for 66 valid")
            return True
        else:
            print("❌ QUALITY TOO LOW (<70% valid)")
            print("   → Claude generation not reliable enough")
            print("   → Recommend: Proceed with 34 samples + theory")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Test Claude sample generation"""

    success = await generate_samples_with_claude(num_samples=10)

    print()
    print("=" * 80)
    if success:
        print("✅ TEST SUCCESSFUL")
        print()
        print("Next steps:")
        print("  1. Review: claude_generated_samples_test.json")
        print("  2. If quality good, generate remaining 56 samples")
        print("  3. Combine with 34 mirror samples")
        print("  4. Run evaluation on 100 total")
    else:
        print("❌ TEST FAILED - Claude generation not reliable")
        print()
        print("Recommendation: Proceed with 34 samples")
        print("  → Move to theory development")
        print("  → 90% ICSE acceptance, 50% ICML")


if __name__ == "__main__":
    asyncio.run(main())
