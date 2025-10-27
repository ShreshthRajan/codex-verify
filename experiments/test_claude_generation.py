"""
Test Claude patch generation on 5 samples before running full 300.
Validates: API connection, patch generation, evaluation pipeline.
"""

import asyncio
import sys
import os

# Modify the import
sys.path.insert(0, os.path.dirname(__file__))

from generate_claude_patches import ClaudePatchGenerator


async def test_generation():
    """Test on 5 samples"""

    print("🧪 TESTING CLAUDE PATCH GENERATION (5 samples)")
    print("=" * 80)
    print()

    try:
        # Create generator with just 5 samples
        generator = ClaudePatchGenerator(sample_size=5)

        print("✅ Generator initialized successfully")
        print(f"🤖 Model: {generator.model}")
        print(f"📊 Testing with: 5 samples")
        print()

        # Run generation
        results = await generator.run_generation_and_evaluation()

        if 'error' in results:
            print(f"\n❌ Test failed: {results['error']}")
            return False

        # Validate results
        print("\n" + "=" * 80)
        print("✅ TEST PASSED!")
        print("=" * 80)
        print()
        print(f"Generated: {results['total_samples']} patches")
        print(f"Time: {results['execution_time']:.1f} seconds")
        print()
        print("🎯 Ready to run full 300-sample generation")
        print()
        print("To run full generation:")
        print("  python generate_claude_patches.py")
        print()

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_generation())
    sys.exit(0 if success else 1)
