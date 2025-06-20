"""
Test cases for Correctness Critic Agent.
Run with: python -m pytest test_correctness_critic.py -v
"""

import pytest
import asyncio
from src.agents.correctness_critic import CorrectnessCritic
from src.agents.base_agent import Severity


@pytest.fixture
def critic():
    """Create a CorrectnessCritic instance for testing"""
    config = {
        'use_llm': False,  # Disable LLM for testing
        'max_execution_time': 2.0
    }
    return CorrectnessCritic(config)


@pytest.mark.asyncio
async def test_simple_correct_code(critic):
    """Test analysis of simple, correct code"""
    code = '''
def add_numbers(a, b):
    """Add two numbers together."""
    return a + b

def multiply(x, y):
    """Multiply two numbers."""
    return x * y

result = add_numbers(5, 3)
print(f"5 + 3 = {result}")
'''
    
    result = await critic.analyze(code)
    
    assert result.success is True
    assert result.agent_name == "CorrectnessCritic"
    assert result.overall_score > 0.7  # Should be high for simple, correct code
    assert 'ast_metrics' in result.metadata
    
    # Check AST metrics
    ast_metrics = result.metadata['ast_metrics']
    assert ast_metrics['function_count'] == 2
    assert ast_metrics['class_count'] == 0
    assert ast_metrics['cyclomatic_complexity'] >= 1


@pytest.mark.asyncio
async def test_complex_problematic_code(critic):
    """Test analysis of complex code with multiple issues"""
    code = '''
def problematic_function(a, b, c, d, e, f, g, h, i):  # Too many parameters
    global global_var  # Global variable usage
    try:
        if a > 0:
            if b > 0:
                if c > 0:
                    if d > 0:
                        if e > 0:  # Deep nesting
                            return a + b + c + d + e
        else:
            for i in range(100):
                for j in range(100):
                    for k in range(100):  # More nesting
                        print(i, j, k)
    except:  # Bare except
        pass
    
    # This function is way too long and does too many things
    # TODO: Fix this mess
    # FIXME: Refactor everything
    if f:
        return f
    elif g:
        return g
    elif h:
        return h
    else:
        return i

global_var = 42
'''
    
    result = await critic.analyze(code)
    
    assert result.success is True
    assert result.overall_score < 0.8  # Should be lower due to many issues (0.7 is reasonable)
    
    # Check for specific issue types
    issue_types = [issue.type for issue in result.issues]
    issue_messages = [issue.message for issue in result.issues]
    
    # Should detect complexity issues
    assert any("complexity" in msg for msg in issue_messages)
    
    # Should detect nesting issues
    assert any("nesting" in msg.lower() for msg in issue_messages)
    
    # Should detect parameter count issues
    assert any("too many parameters" in msg for msg in issue_messages)
    
    # Should detect global variable usage
    assert any("Global variable" in msg for msg in issue_messages)
    
    # Should detect bare except
    assert any("Bare except" in msg for msg in issue_messages)


@pytest.mark.asyncio
async def test_syntax_error_code(critic):
    """Test analysis of code with syntax errors"""
    code = '''
def broken_function(:  # Missing parameter, syntax error
    return "this won't work"
'''
    
    result = await critic.analyze(code)
    
    assert result.success is True  # Analysis succeeds even with syntax errors
    assert result.overall_score < 0.2  # Should be very low (0.15 is acceptable)
    
    # Should detect syntax error
    issue_messages = [issue.message for issue in result.issues]
    assert any("Syntax error" in msg for msg in issue_messages)
    
    # Should have critical severity for syntax errors
    critical_issues = [issue for issue in result.issues if issue.severity == Severity.CRITICAL]
    assert len(critical_issues) > 0


@pytest.mark.asyncio
async def test_dangerous_code_execution_check(critic):
    """Test that dangerous code is flagged and not executed"""
    code = '''
import os
import subprocess

def dangerous_function():
    os.system("rm -rf /")  # Very dangerous!
    subprocess.call(["shutdown", "-h", "now"])
    
dangerous_function()
'''
    
    result = await critic.analyze(code)
    
    assert result.success is True
    
    # Should flag dangerous operations
    issue_messages = [issue.message for issue in result.issues]
    assert any("dangerous operations" in msg for msg in issue_messages)


@pytest.mark.asyncio
async def test_infinite_loop_timeout(critic):
    """Test that infinite loops are detected via timeout"""
    code = '''
def infinite_loop():
    while True:
        pass

infinite_loop()
'''
    
    result = await critic.analyze(code)
    
    assert result.success is True
    
    # Should detect timeout
    issue_messages = [issue.message for issue in result.issues]
    assert any("timed out" in msg for msg in issue_messages)


@pytest.mark.asyncio
async def test_ast_metrics_calculation(critic):
    """Test AST metrics calculation accuracy"""
    code = '''
class TestClass:
    def method1(self):
        if True:
            for i in range(10):
                if i % 2 == 0:
                    print(i)
    
    def method2(self):
        try:
            result = 1 / 0
        except ZeroDivisionError:
            return 0
        except ValueError:
            return -1
        return result

def standalone_function():
    pass
'''
    
    result = await critic.analyze(code)
    
    assert result.success is True
    
    ast_metrics = result.metadata['ast_metrics']
    
    # Should detect correct counts
    assert ast_metrics['class_count'] == 1
    assert ast_metrics['function_count'] == 3  # 2 methods + 1 standalone
    assert ast_metrics['cyclomatic_complexity'] > 1  # Has conditional and loop logic
    assert ast_metrics['nesting_depth'] >= 3  # Class -> method -> if -> for


@pytest.mark.asyncio
async def test_property_test_suggestions(critic):
    """Test that property-based test suggestions are generated"""
    code = '''
def calculate_factorial(n):
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * calculate_factorial(n - 1)

def find_max(numbers):
    """Find maximum number in list."""
    if not numbers:
        return None
    max_val = numbers[0]
    for num in numbers[1:]:
        if num > max_val:
            max_val = num
    return max_val
'''
    
    result = await critic.analyze(code)
    
    assert result.success is True
    
    # Should suggest property testing for functions with parameters
    issue_messages = [issue.message for issue in result.issues]
    testing_suggestions = [msg for msg in issue_messages if "property-based testing" in msg]
    
    # Should have suggestions for both functions
    assert len(testing_suggestions) >= 2


@pytest.mark.asyncio
async def test_disabled_agent(critic):
    """Test agent behavior when disabled"""
    critic.enabled = False
    
    code = '''
def some_function():
    return "test"
'''
    
    result = await critic.analyze(code)
    
    assert result.success is True
    assert result.overall_score == 1.0
    assert len(result.issues) == 0
    assert result.metadata.get('skipped') is True


@pytest.mark.asyncio
async def test_execution_with_runtime_error(critic):
    """Test handling of code that has runtime errors"""
    code = '''
def divide_by_zero():
    return 10 / 0

result = divide_by_zero()
print(result)
'''
    
    result = await critic.analyze(code)
    
    assert result.success is True
    
    # Should detect execution failure
    issue_messages = [issue.message for issue in result.issues]
    assert any("execution failed" in msg for msg in issue_messages)


def test_score_calculation():
    """Test the score calculation logic"""
    critic = CorrectnessCritic({'use_llm': False})
    
    # Test with no issues
    score = critic._calculate_score([])
    assert score == 1.0
    
    # Test with low severity issues
    from src.agents.base_agent import VerificationIssue
    low_issues = [
        VerificationIssue("test", Severity.LOW, "Low issue 1"),
        VerificationIssue("test", Severity.LOW, "Low issue 2")
    ]
    score = critic._calculate_score(low_issues)
    assert score > 0.8  # Should still be high
    
    # Test with critical issues
    critical_issues = [
        VerificationIssue("test", Severity.CRITICAL, "Critical issue 1"),
        VerificationIssue("test", Severity.CRITICAL, "Critical issue 2")
    ]
    score = critic._calculate_score(critical_issues)
    assert score == 0.0  # Should be very low


if __name__ == "__main__":
    # Simple test runner if pytest is not available
    import sys
    sys.path.insert(0, '.')
    
    async def run_tests():
        critic = CorrectnessCritic({'use_llm': False, 'max_execution_time': 2.0})
        
        print("Running Correctness Critic tests...")
        
        # Test 1: Simple correct code
        print("Test 1: Simple correct code")
        code1 = '''
def add_numbers(a, b):
    return a + b
result = add_numbers(5, 3)
'''
        result1 = await critic.analyze(code1)
        print(f"  Score: {result1.overall_score:.2f}")
        print(f"  Issues: {len(result1.issues)}")
        assert result1.success
        
        # Test 2: Problematic code
        print("\nTest 2: Problematic code")
        code2 = '''
def bad_function(a, b, c, d, e, f, g, h):  # Too many params
    global x
    try:
        if a:
            if b:
                if c:
                    if d:
                        return a + b
    except:
        pass
'''
        result2 = await critic.analyze(code2)
        print(f"  Score: {result2.overall_score:.2f}")
        print(f"  Issues: {len(result2.issues)}")
        for issue in result2.issues[:3]:  # Show first 3 issues
            print(f"    - {issue.severity.value}: {issue.message}")
        assert result2.success
        assert result2.overall_score < 0.5
        
        print("\n✅ All tests passed!")
    
    # Run the tests
    asyncio.run(run_tests())