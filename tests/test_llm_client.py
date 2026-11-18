#!/usr/bin/env python3
"""
Test LLM Client - Validation Suite

Tests Claude client with real API calls on sample code.
Requires ANTHROPIC_API_KEY environment variable.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.claude_client import ClaudeClient


# Test samples (one per agent type)
TEST_SAMPLES = {
    'correctness': '''
def divide_numbers(a, b):
    # BUG: No zero division check
    return a / b

def process_list(items):
    # BUG: No None/empty check
    return items[0]
''',

    'security': '''
import sqlite3

def get_user(username):
    conn = sqlite3.connect('users.db')
    # BUG: SQL injection
    query = f"SELECT * FROM users WHERE name = '{username}'"
    return conn.execute(query).fetchone()
''',

    'performance': '''
def find_duplicates(numbers):
    # BUG: O(n²) complexity
    duplicates = []
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            if numbers[i] == numbers[j]:
                duplicates.append(numbers[i])
    return duplicates
''',

    'style': '''
def f(x,y,z):
    # BUG: Poor naming, no docstring
    return x+y*z
'''
}


async def test_agent(client: ClaudeClient, agent_type: str):
    """Test one agent type"""
    print(f"\n{'='*80}")
    print(f"Testing {agent_type.upper()} Agent")
    print('='*80)

    code = TEST_SAMPLES[agent_type]
    print(f"Code:\n{code[:200]}...")

    try:
        response = await client.analyze_code(code, agent_type)

        print(f"\n✓ Response received:")
        print(f"  Issues found: {len(response.issues)}")
        print(f"  Cost: ${response.cost_usd:.4f}")
        print(f"  Latency: {response.latency_seconds:.2f}s")
        print(f"  Tokens: {response.tokens_used}")

        if response.issues:
            print(f"\n  Sample issues:")
            for issue in response.issues[:2]:
                print(f"    - [{issue.get('severity')}] {issue.get('type')}: {issue.get('message')[:60]}...")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


async def main():
    """Run all tests"""
    print("="*80)
    print("LLM CLIENT VALIDATION TESTS")
    print("="*80)

    # Check API key
    import os
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("\n❌ ANTHROPIC_API_KEY not set")
        print("\nSet it:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        return

    # Initialize client
    client = ClaudeClient()
    print(f"\n✓ Claude client initialized")
    print(f"  Model: {client.MODEL}")

    # Test each agent type
    results = {}
    for agent_type in ['correctness', 'security', 'performance', 'style']:
        success = await test_agent(client, agent_type)
        results[agent_type] = success

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for agent_type, success in results.items():
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"  {agent_type:<15} {status}")

    # Stats
    stats = client.get_stats()
    print(f"\n📊 Total Stats:")
    print(f"  Requests: {stats['total_requests']}")
    print(f"  Total cost: ${stats['total_cost_usd']:.4f}")
    print(f"  Total tokens: {stats['total_tokens']}")

    if all(results.values()):
        print("\n✅ ALL TESTS PASSED - LLM client is ready")
    else:
        print("\n⚠️  Some tests failed - check API key and connectivity")


if __name__ == '__main__':
    asyncio.run(main())
