#!/usr/bin/env python3
"""
Demo script to test Agent 1 (Correctness Critic) functionality.
Place this in the project root and run: python demo_test_agent1.py
"""

import sys
import os
import asyncio

# Add project root to path - works from any directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if 'tests' in current_dir:
    # If running from tests directory, go up to project root
    project_root = os.path.dirname(os.path.dirname(current_dir))

sys.path.insert(0, project_root)

try:
    from src.agents.correctness_critic import CorrectnessCritic
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[:3]}...")
    print("\n💡 Make sure you're running from the project root directory:")
    print("   cd codex-verify")
    print("   python demo_test_agent1.py")
    sys.exit(1)


async def demo_correctness_critic():
    """Demonstrate the Correctness Critic agent capabilities"""
    
    print("🔍 CODEX VERIFICATION FRAMEWORK - AGENT 1 DEMO")
    print("=" * 60)
    
    # Initialize the agent
    config = {
        'use_llm': False,  # Set to True if you have API keys
        'max_execution_time': 3.0,
        'enabled': True
    }
    
    critic = CorrectnessCritic(config)
    print(f"✅ Initialized {critic.name}")
    print()
    
    # Test cases
    test_cases = [
        {
            'name': 'Clean, Simple Code',
            'code': '''
def calculate_circle_area(radius):
    """Calculate the area of a circle."""
    import math
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    return math.pi * radius ** 2

def format_area(area):
    """Format area for display."""
    return f"Area: {area:.2f} square units"

# Example usage
radius = 5.0
area = calculate_circle_area(radius)
print(format_area(area))
'''
        },
        {
            'name': 'Problematic Code with Multiple Issues',
            'code': '''
def messy_function(a, b, c, d, e, f, g, h, i, j):  # Too many parameters
    global problematic_global  # Global usage
    try:
        if a > 0:
            if b > 0:
                if c > 0:
                    if d > 0:
                        if e > 0:
                            if f > 0:  # Excessive nesting
                                return a + b + c + d + e + f
                            else:
                                for x in range(1000):
                                    for y in range(1000):  # Nested loops
                                        print(x * y)
        elif g:
            return g * 2
        elif h:
            return h * 3
        elif i:
            return i * 4
        else:
            return j * 5
    except:  # Bare except clause
        pass
    
    # TODO: This function needs major refactoring
    # FIXME: Too complex and unclear

problematic_global = 42
'''
        },
        {
            'name': 'Code with Syntax Error',
            'code': '''
def broken_function(:  # Missing parameter name
    return "This has a syntax error"

def another_function()
    return "Missing colon"  # Missing colon in function definition
'''
        },
        {
            'name': 'Code with Runtime Error',
            'code': '''
def divide_numbers(a, b):
    """Divide two numbers."""
    return a / b

def main():
    result = divide_numbers(10, 0)  # Division by zero
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
'''
        }
    ]
    
    # Run analysis on each test case
    for i, test_case in enumerate(test_cases, 1):
        print(f"📋 TEST CASE {i}: {test_case['name']}")
        print("-" * 50)
        
        try:
            result = await critic.analyze(test_case['code'])
            
            # Display results
            print(f"✅ Analysis Status: {'SUCCESS' if result.success else 'FAILED'}")
            print(f"🎯 Overall Score: {result.overall_score:.2f}/1.00")
            print(f"⏱️  Execution Time: {result.execution_time:.3f}s")
            print(f"🔍 Issues Found: {len(result.issues)}")
            
            # Display AST metrics if available
            if 'ast_metrics' in result.metadata:
                metrics = result.metadata['ast_metrics']
                print(f"📊 AST Metrics:")
                print(f"   • Cyclomatic Complexity: {metrics['cyclomatic_complexity']}")
                print(f"   • Nesting Depth: {metrics['nesting_depth']}")
                print(f"   • Functions: {metrics['function_count']}")
                print(f"   • Classes: {metrics['class_count']}")
                print(f"   • Lines of Code: {metrics['line_count']}")
            
            # Display issues by severity
            if result.issues:
                print(f"🚨 Issues by Severity:")
                for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                    severity_issues = [issue for issue in result.issues 
                                     if issue.severity.value.upper() == severity]
                    if severity_issues:
                        print(f"   {severity}: {len(severity_issues)} issues")
                        for issue in severity_issues[:2]:  # Show first 2 of each severity
                            print(f"      • {issue.message}")
                        if len(severity_issues) > 2:
                            print(f"      ... and {len(severity_issues) - 2} more")
            else:
                print("✨ No issues found!")
            
            # Display semantic analysis if available
            if 'semantic_analysis' in result.metadata:
                semantic = result.metadata['semantic_analysis']
                print(f"🧠 Semantic Analysis:")
                print(f"   • Logic Score: {semantic['logic_score']:.2f}")
                print(f"   • Clarity Score: {semantic['clarity_score']:.2f}")
            
            print()
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            print()
    
    print("🎉 DEMO COMPLETED!")
    print("\n💡 Next Steps:")
    print("1. Add API keys to config for LLM-powered semantic analysis")
    print("2. Test with your own code samples")
    print("3. Integration with other agents in Phase 2")


if __name__ == "__main__":
    # Check current working directory
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📂 Script location: {os.path.dirname(os.path.abspath(__file__))}")
    
    # Check dependencies
    missing_deps = []
    
    try:
        import aiohttp
    except ImportError:
        missing_deps.append('aiohttp')
    
    try:
        import tree_sitter
    except ImportError:
        print("⚠️  Warning: tree-sitter not available (optional)")
    
    try:
        import hypothesis
    except ImportError:
        print("⚠️  Warning: hypothesis not available (optional)")
    
    if missing_deps:
        print(f"❌ Missing required dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install -r requirements.txt")
        sys.exit(1)
    
    # Run the demo
    asyncio.run(demo_correctness_critic())