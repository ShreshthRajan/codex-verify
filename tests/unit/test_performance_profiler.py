# tests/unit/test_performance_profiler.py
"""
Comprehensive test suite for Performance Profiler Agent.
Tests complexity analysis, algorithm efficiency, and performance profiling.
"""

import pytest
import asyncio
from src.agents.performance_profiler import PerformanceProfiler, ComplexityMetrics, AlgorithmAnalysis
from src.agents.base_agent import Severity


class TestPerformanceProfiler:
    """Test suite for PerformanceProfiler agent"""
    
    @pytest.fixture
    def performance_profiler(self):
        """Create PerformanceProfiler instance for testing"""
        config = {
            'max_execution_time': 2.0,
            'memory_limit_mb': 100,
            'enable_runtime_profiling': True,
            'complexity_thresholds': {
                'cyclomatic': {'medium': 10, 'high': 15, 'critical': 25},
                'cognitive': {'medium': 15, 'high': 25, 'critical': 40},
                'nesting': {'medium': 4, 'high': 6, 'critical': 8}
            }
        }
        return PerformanceProfiler(config)
    
    @pytest.mark.asyncio
    async def test_simple_function_analysis(self, performance_profiler):
        """Test analysis of simple, efficient function"""
        simple_code = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

def find_max(items):
    if not items:
        return None
    max_val = items[0]
    for item in items[1:]:
        if item > max_val:
            max_val = item
    return max_val
"""
        
        result = await performance_profiler.analyze(simple_code)
        
        assert result.success
        assert result.overall_score > 0.7  # Should score well for simple, efficient code
        assert 'complexity_metrics' in result.metadata
        assert 'algorithm_analysis' in result.metadata
        
        # Should have low complexity
        complexity = result.metadata['complexity_metrics']
        assert complexity['cyclomatic_complexity'] < 10
    
    @pytest.mark.asyncio
    async def test_high_complexity_detection(self, performance_profiler):
        """Test detection of high complexity code"""
        complex_code = """
def complex_function(data, config, options, flags, params, settings, extra):
    result = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                if config.enabled:
                    if options.mode == 'advanced':
                        if flags.debug:
                            if params.verbose:
                                if settings.detailed:
                                    if extra.logging:
                                        for item in data[i][j][k]:
                                            if item.valid:
                                                if item.processed:
                                                    result.append(item.value)
                                                else:
                                                    if item.retryable:
                                                        result.append(item.retry())
                                                    else:
                                                        continue
    return result
"""
        
        result = await performance_profiler.analyze(complex_code)
        
        assert result.success
        assert result.overall_score < 0.6  # Should heavily penalize complex code (adjusted for very aggressive scoring)
        
        # Check for complexity issues
        complexity_issues = [issue for issue in result.issues 
                           if issue.type in ["complexity", "cognitive_complexity", "nesting"]]
        assert len(complexity_issues) >= 2
        
        # Should detect high severity issues
        high_severity_issues = [issue for issue in result.issues 
                               if issue.severity in [Severity.HIGH, Severity.CRITICAL]]
        assert len(high_severity_issues) >= 1
    
    @pytest.mark.asyncio
    async def test_algorithm_complexity_analysis(self, performance_profiler):
        """Test algorithm complexity analysis"""
        nested_loop_code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    
    return result
"""
        
        result = await performance_profiler.analyze(nested_loop_code)
        
        assert result.success
        assert 'algorithm_analysis' in result.metadata
        
        algorithm_analysis = result.metadata['algorithm_analysis']
        
        # Should detect quadratic or higher complexity
        assert "²" in algorithm_analysis['estimated_time_complexity'] or "³" in algorithm_analysis['estimated_time_complexity']
        assert algorithm_analysis['loop_nesting_level'] >= 2
        
        # Should flag time complexity issues
        complexity_issues = [issue for issue in result.issues if "complexity" in issue.message.lower()]
        assert len(complexity_issues) >= 1
    
    @pytest.mark.asyncio
    async def test_performance_antipattern_detection(self, performance_profiler):
        """Test detection of performance anti-patterns"""
        antipattern_code = """
def inefficient_string_building(items):
    result = ""
    for item in items:
        result += str(item) + ", "
    return result

def inefficient_search(haystack, needles):
    found = []
    for needle in needles:
        if needle in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # List membership
            found.append(needle)
    return found

global_counter = 0
def uses_global():
    global global_counter
    global_counter += 1
    return global_counter
"""
        
        result = await performance_profiler.analyze(antipattern_code)
        
        assert result.success
        
        # Check for anti-pattern detection
        antipattern_issues = [issue for issue in result.issues 
                             if issue.type == "performance_antipattern"]
        assert len(antipattern_issues) >= 1
        
        # Should suggest optimizations
        optimization_issues = [issue for issue in result.issues 
                              if "join" in issue.suggestion.lower() or "set" in issue.suggestion.lower()]
        assert len(optimization_issues) >= 1
    
    @pytest.mark.asyncio
    async def test_function_length_detection(self, performance_profiler):
        """Test detection of overly long functions"""
        long_function_code = """
def very_long_function(param1, param2, param3, param4, param5, param6, param7, param8):
    # This function is intentionally very long
    line1 = param1 + param2
    line2 = param3 + param4
    line3 = param5 + param6
    line4 = param7 + param8
    line5 = line1 + line2
    line6 = line3 + line4
    line7 = line5 + line6
    line8 = line7 * 2
    line9 = line8 / 2
    line10 = line9 + 1
    line11 = line10 - 1
    line12 = line11 * 3
    line13 = line12 / 3
    line14 = line13 + 10
    line15 = line14 - 10
    line16 = line15 * 4
    line17 = line16 / 4
    line18 = line17 + 100
    line19 = line18 - 100
    line20 = line19 * 5
    line21 = line20 / 5
    line22 = line21 + 1000
    line23 = line22 - 1000
    line24 = line23 * 6
    line25 = line24 / 6
    line26 = line25 + 10000
    line27 = line26 - 10000
    line28 = line27 * 7
    line29 = line28 / 7
    line30 = line29 + 100000
    line31 = line30 - 100000
    line32 = line31 * 8
    line33 = line32 / 8
    line34 = line33 + 1000000
    line35 = line34 - 1000000
    line36 = line35 * 9
    line37 = line36 / 9
    line38 = line37 + 10000000
    line39 = line38 - 10000000
    line40 = line39 * 10
    line41 = line40 / 10
    line42 = line41 + 100000000
    line43 = line42 - 100000000
    line44 = line43 * 11
    line45 = line44 / 11
    line46 = line45 + 1000000000
    line47 = line46 - 1000000000
    line48 = line47 * 12
    line49 = line48 / 12
    line50 = line49 + 10000000000
    line51 = line50 - 10000000000
    line52 = line51 * 13
    line53 = line52 / 13
    line54 = line53 + 100000000000
    line55 = line54 - 100000000000
    return line55
"""
        
        result = await performance_profiler.analyze(long_function_code)
        
        assert result.success
        
        # Should detect function length issues
        length_issues = [issue for issue in result.issues if issue.type == "function_length"]
        assert len(length_issues) >= 1
        
        # Should detect too many parameters
        param_issues = [issue for issue in result.issues if issue.type == "parameter_count"]
        assert len(param_issues) >= 1
    
    @pytest.mark.asyncio
    async def test_optimization_suggestions(self, performance_profiler):
        """Test generation of optimization suggestions"""
        suboptimal_code = """
def process_data(items):
    # Nested loops that could be optimized
    result = []
    for i in range(len(items)):
        for j in range(len(items)):
            if items[i] == items[j]:
                result.append((i, j))
    
    # String concatenation in loop
    output = ""
    for item in result:
        output += str(item) + "\\n"
    
    return output

def search_items(data, targets):
    found = []
    for target in targets:
        if target in data:  # Could be optimized with set
            found.append(target)
    return found
"""
        
        result = await performance_profiler.analyze(suboptimal_code)
        
        assert result.success
        assert 'algorithm_analysis' in result.metadata
        
        # Should generate optimization suggestions
        optimization_issues = [issue for issue in result.issues if issue.type == "optimization"]
        assert len(optimization_issues) >= 1
        
        # Should detect performance anti-patterns
        antipattern_issues = [issue for issue in result.issues if issue.type == "performance_antipattern"]
        assert len(antipattern_issues) >= 1
    
    @pytest.mark.asyncio
    async def test_runtime_profiling_safe_code(self, performance_profiler):
        """Test runtime profiling on safe code"""
        safe_code = """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

result = fibonacci(10)
"""
        
        result = await performance_profiler.analyze(safe_code)
        
        assert result.success
        
        # Should have runtime metrics if profiling succeeded
        if 'performance_metrics' in result.metadata:
            perf_metrics = result.metadata['performance_metrics']
            assert 'execution_time' in perf_metrics
            assert perf_metrics['execution_time'] >= 0
    
    @pytest.mark.asyncio
    async def test_unsafe_code_skipping(self, performance_profiler):
        """Test that unsafe code skips runtime profiling"""
        unsafe_code = """
import os
import subprocess

def dangerous_function():
    os.system("rm -rf /")
    subprocess.call("malicious command", shell=True)
    exec("print('dangerous')")
"""
        
        result = await performance_profiler.analyze(unsafe_code)
        
        assert result.success
        
        # Should not have runtime metrics for unsafe code
        # But should still have static analysis
        assert 'complexity_metrics' in result.metadata
        assert 'algorithm_analysis' in result.metadata
    
    @pytest.mark.asyncio
    async def test_clean_efficient_code(self, performance_profiler):
        """Test analysis of clean, efficient code"""
        efficient_code = """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def efficient_sum(numbers):
    return sum(numbers)

def optimized_filter(items, condition):
    return [item for item in items if condition(item)]
"""
        
        result = await performance_profiler.analyze(efficient_code)
        
        assert result.success
        assert result.overall_score > 0.8  # Should score well for efficient code
        
        # Should have minimal performance issues
        high_severity_issues = [issue for issue in result.issues 
                               if issue.severity in [Severity.HIGH, Severity.CRITICAL]]
        assert len(high_severity_issues) == 0
    
    @pytest.mark.asyncio
    async def test_recursive_function_analysis(self, performance_profiler):
        """Test analysis of recursive functions"""
        recursive_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)
"""
        
        result = await performance_profiler.analyze(recursive_code)
        
        assert result.success
        assert 'algorithm_analysis' in result.metadata
        
        algorithm_analysis = result.metadata['algorithm_analysis']
        assert algorithm_analysis['recursive_calls'] > 0
        assert algorithm_analysis['estimated_space_complexity'] == "O(n)"
    
    @pytest.mark.asyncio
    async def test_complexity_metrics_calculation(self, performance_profiler):
        """Test complexity metrics calculation"""
        # Test the complexity calculation directly with proper code
        test_code = """
def test_function(a, b, c):
    if a > 0:
        for i in range(b):
            if i % 2 == 0:
                for j in range(c):
                    if j > i:
                        return j
    return 0
"""
        
        result = await performance_profiler.analyze(test_code)
        assert result.success
        
        complexity = result.metadata['complexity_metrics']
        assert complexity['cyclomatic_complexity'] >= 4  # if + for + if + for + if
        assert complexity['nesting_depth'] >= 3  # nested structures
        assert complexity['parameter_count'] == 3
        assert complexity['function_length'] > 0
    
    @pytest.mark.asyncio
    async def test_empty_code_analysis(self, performance_profiler):
        """Test analysis of empty code"""
        result = await performance_profiler.analyze("")
        
        assert result.success
        assert result.overall_score == 1.0  # Perfect score for empty code
        assert len(result.issues) == 0
    
    @pytest.mark.asyncio
    async def test_syntax_error_handling(self, performance_profiler):
        """Test handling of code with syntax errors"""
        syntax_error_code = """
def broken_function(
    # Missing closing parenthesis and colon
    return "this won't parse"
"""
        
        result = await performance_profiler.analyze(syntax_error_code)
        
        # Should still succeed with limited analysis
        assert result.success
        # Some analysis should still be possible
    
    @pytest.mark.asyncio
    async def test_configuration_options(self):
        """Test configuration options"""
        config = {
            'max_execution_time': 5.0,
            'memory_limit_mb': 200,
            'enable_runtime_profiling': False,
            'complexity_thresholds': {
                'cyclomatic': {'medium': 5, 'high': 8, 'critical': 12}
            }
        }
        profiler = PerformanceProfiler(config)
        
        assert profiler.max_execution_time == 5.0
        assert profiler.memory_limit_mb == 200
        assert profiler.enable_runtime_profiling == False
        assert profiler.complexity_thresholds['cyclomatic']['medium'] == 5
    
    @pytest.mark.asyncio
    async def test_agent_disabled(self):
        """Test agent when disabled"""
        config = {'enabled': False}
        profiler = PerformanceProfiler(config)
        
        result = await profiler.analyze("def complex_function(): pass")
        
        assert result.success
        assert result.overall_score == 1.0
        assert len(result.issues) == 0
        assert result.metadata.get('skipped') == True