# src/agents/performance_profiler.py
"""
Performance Profiler Agent - Algorithm efficiency and resource usage analysis.
Analyzes code for performance issues, complexity metrics, and optimization opportunities.
"""

import ast
import re
import time
import sys
import tempfile
import os
import subprocess
import tracemalloc
from typing import Dict, List, Any, Optional, Tuple, Set
import asyncio
from dataclasses import dataclass
from collections import defaultdict
import math

try:
    import cProfile
    import pstats
    import io
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

from .base_agent import BaseAgent, AgentResult, VerificationIssue, Severity


@dataclass
class ComplexityMetrics:
    """Code complexity metrics"""
    cyclomatic_complexity: int
    cognitive_complexity: int
    nesting_depth: int
    function_length: int
    parameter_count: int
    return_statements: int
    loop_count: int
    conditional_count: int


@dataclass
class AlgorithmAnalysis:
    """Algorithm efficiency analysis"""
    estimated_time_complexity: str
    estimated_space_complexity: str
    loop_nesting_level: int
    recursive_calls: int
    data_structure_efficiency: Dict[str, str]
    potential_optimizations: List[str]


@dataclass
class PerformanceMetrics:
    """Runtime performance metrics"""
    execution_time: float
    memory_usage: float
    peak_memory: float
    function_call_count: int
    io_operations: int
    performance_score: float


@dataclass
class ResourceAnalysis:
    """Resource usage analysis"""
    memory_efficiency: float
    cpu_efficiency: float
    io_efficiency: float
    scalability_concerns: List[str]
    bottleneck_locations: List[Tuple[int, str]]


class PerformanceProfiler(BaseAgent):
    """
    Agent 3: Performance Profiler
    
    Performs comprehensive performance analysis including:
    - Complexity calculation (cyclomatic, cognitive, algorithmic)
    - Runtime benchmarking and memory profiling
    - Algorithm efficiency analysis
    - Resource usage optimization suggestions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("PerformanceProfiler", config)
        self.max_execution_time = self.config.get('max_execution_time', 2.0)
        self.memory_limit_mb = self.config.get('memory_limit_mb', 100)
        self.enable_runtime_profiling = self.config.get('enable_runtime_profiling', True)
        self.complexity_thresholds = self.config.get('complexity_thresholds', {
            'cyclomatic': {'medium': 10, 'high': 15, 'critical': 25},
            'cognitive': {'medium': 15, 'high': 25, 'critical': 40},
            'nesting': {'medium': 4, 'high': 6, 'critical': 8}
        })
    
    async def _analyze_implementation(self, code: str, context: Dict[str, Any]) -> AgentResult:
        """Main implementation of performance analysis"""
        issues = []
        metadata = {}
        
        # 1. Complexity Analysis
        complexity_metrics = self._analyze_complexity(code)
        complexity_issues = self._extract_complexity_issues(complexity_metrics)
        issues.extend(complexity_issues)
        metadata['complexity_metrics'] = complexity_metrics.__dict__
        
        # 2. Algorithm Analysis
        algorithm_analysis = self._analyze_algorithms(code)
        algorithm_issues = self._extract_algorithm_issues(algorithm_analysis)
        issues.extend(algorithm_issues)
        metadata['algorithm_analysis'] = algorithm_analysis.__dict__
        
        # 3. Static Performance Analysis
        static_issues = self._static_performance_analysis(code)
        issues.extend(static_issues)
        
        # 4. Runtime Profiling (if enabled and safe)
        if self.enable_runtime_profiling and self._is_safe_for_execution(code):
            try:
                performance_metrics = await self._runtime_profiling(code)
                runtime_issues = self._extract_runtime_issues(performance_metrics)
                issues.extend(runtime_issues)
                metadata['performance_metrics'] = performance_metrics.__dict__
            except Exception as e:
                issues.append(VerificationIssue(
                    type="profiling_error",
                    severity=Severity.LOW,
                    message=f"Runtime profiling failed: {str(e)}",
                    suggestion="Manual performance testing recommended"
                ))
        
        # 5. Resource Analysis
        resource_analysis = self._analyze_resource_usage(code)
        resource_issues = self._extract_resource_issues(resource_analysis)
        issues.extend(resource_issues)
        metadata['resource_analysis'] = resource_analysis.__dict__
        
        # Calculate overall performance score
        overall_score = self._calculate_performance_score(issues, metadata)
        
        return AgentResult(
            agent_name=self.name,
            execution_time=0.0,  # Will be set by base class
            overall_score=overall_score,
            issues=issues,
            metadata=metadata
        )
    
    def _analyze_complexity(self, code: str) -> ComplexityMetrics:
        """Analyze code complexity metrics"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ComplexityMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        # Find all functions for analysis
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        
        if not functions:
            # Analyze the module level if no functions
            return self._calculate_complexity_for_node(tree)
        
        # Analyze the most complex function
        max_complexity = ComplexityMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        for func in functions:
            complexity = self._calculate_complexity_for_node(func)
            if complexity.cyclomatic_complexity > max_complexity.cyclomatic_complexity:
                max_complexity = complexity
        
        return max_complexity
    
    def _calculate_complexity_for_node(self, node: ast.AST) -> ComplexityMetrics:
        """Calculate complexity metrics for an AST node"""
        cyclomatic = 1  # Base complexity
        cognitive = 0
        nesting_depth = 0
        function_length = 0
        parameter_count = 0
        return_statements = 0
        loop_count = 0
        conditional_count = 0
        
        # Calculate function-specific metrics if it's a function
        if isinstance(node, ast.FunctionDef):
            parameter_count = len(node.args.args)
            if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                function_length = node.end_lineno - node.lineno
        
        # Walk through all nodes and calculate metrics
        for child in ast.walk(node):
            # Cyclomatic complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                cyclomatic += 1
                conditional_count += 1
            elif isinstance(child, ast.ExceptHandler):
                cyclomatic += 1
            elif isinstance(child, ast.With):
                cyclomatic += 1
            elif isinstance(child, ast.BoolOp):
                cyclomatic += len(child.values) - 1
            
            # Loop counting
            if isinstance(child, (ast.For, ast.While, ast.AsyncFor)):
                loop_count += 1
            
            # Return statements
            if isinstance(child, ast.Return):
                return_statements += 1
        
        # Calculate nesting depth
        nesting_depth = self._calculate_nesting_depth(node)
        
        # Calculate cognitive complexity (simplified)
        cognitive = self._calculate_cognitive_complexity(node)
        
        return ComplexityMetrics(
            cyclomatic_complexity=cyclomatic,
            cognitive_complexity=cognitive,
            nesting_depth=nesting_depth,
            function_length=function_length,
            parameter_count=parameter_count,
            return_statements=return_statements,
            loop_count=loop_count,
            conditional_count=conditional_count
        )
    
    def _calculate_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth"""
        max_depth = current_depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                child_depth = self._calculate_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _calculate_cognitive_complexity(self, node: ast.AST, nesting_level: int = 0) -> int:
        """Calculate cognitive complexity (simplified version)"""
        cognitive = 0
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                cognitive += 1 + nesting_level
                cognitive += self._calculate_cognitive_complexity(child, nesting_level + 1)
            elif isinstance(child, ast.BoolOp):
                cognitive += len(child.values) - 1
            elif isinstance(child, ast.ExceptHandler):
                cognitive += 1 + nesting_level
            else:
                cognitive += self._calculate_cognitive_complexity(child, nesting_level)
        
        return cognitive
    
    def _extract_complexity_issues(self, metrics: ComplexityMetrics) -> List[VerificationIssue]:
        """Extract issues from complexity metrics"""
        issues = []
        thresholds = self.complexity_thresholds
        
        # Cyclomatic complexity issues
        if metrics.cyclomatic_complexity >= thresholds['cyclomatic']['critical']:
            issues.append(VerificationIssue(
                type="complexity",
                severity=Severity.CRITICAL,
                message=f"Extremely high cyclomatic complexity: {metrics.cyclomatic_complexity}",
                suggestion="Break function into smaller, focused functions"
            ))
        elif metrics.cyclomatic_complexity >= thresholds['cyclomatic']['high']:
            issues.append(VerificationIssue(
                type="complexity",
                severity=Severity.HIGH,
                message=f"High cyclomatic complexity: {metrics.cyclomatic_complexity}",
                suggestion="Consider refactoring to reduce complexity"
            ))
        elif metrics.cyclomatic_complexity >= thresholds['cyclomatic']['medium']:
            issues.append(VerificationIssue(
                type="complexity",
                severity=Severity.MEDIUM,
                message=f"Moderate cyclomatic complexity: {metrics.cyclomatic_complexity}",
                suggestion="Monitor complexity growth"
            ))
        
        # Cognitive complexity issues
        if metrics.cognitive_complexity >= thresholds['cognitive']['critical']:
            issues.append(VerificationIssue(
                type="cognitive_complexity",
                severity=Severity.CRITICAL,
                message=f"Extremely high cognitive complexity: {metrics.cognitive_complexity}",
                suggestion="Simplify logic and reduce nesting"
            ))
        elif metrics.cognitive_complexity >= thresholds['cognitive']['high']:
            issues.append(VerificationIssue(
                type="cognitive_complexity",
                severity=Severity.HIGH,
                message=f"High cognitive complexity: {metrics.cognitive_complexity}",
                suggestion="Reduce logical complexity and nesting"
            ))
        
        # Nesting depth issues
        if metrics.nesting_depth >= thresholds['nesting']['critical']:
            issues.append(VerificationIssue(
                type="nesting",
                severity=Severity.CRITICAL,
                message=f"Excessive nesting depth: {metrics.nesting_depth}",
                suggestion="Extract nested logic into separate functions"
            ))
        elif metrics.nesting_depth >= thresholds['nesting']['high']:
            issues.append(VerificationIssue(
                type="nesting",
                severity=Severity.HIGH,
                message=f"High nesting depth: {metrics.nesting_depth}",
                suggestion="Consider reducing nesting levels"
            ))
        
        # Function length issues
        if metrics.function_length > 100:
            issues.append(VerificationIssue(
                type="function_length",
                severity=Severity.HIGH,
                message=f"Very long function: {metrics.function_length} lines",
                suggestion="Break function into smaller, focused functions"
            ))
        elif metrics.function_length > 50:
            issues.append(VerificationIssue(
                type="function_length",
                severity=Severity.MEDIUM,
                message=f"Long function: {metrics.function_length} lines",
                suggestion="Consider breaking into smaller functions"
            ))
        
        # Parameter count issues
        if metrics.parameter_count > 10:
            issues.append(VerificationIssue(
                type="parameter_count",
                severity=Severity.HIGH,
                message=f"Too many parameters: {metrics.parameter_count}",
                suggestion="Use data classes or configuration objects"
            ))
        elif metrics.parameter_count > 7:
            issues.append(VerificationIssue(
                type="parameter_count",
                severity=Severity.MEDIUM,
                message=f"Many parameters: {metrics.parameter_count}",
                suggestion="Consider parameter object or builder pattern"
            ))
        
        return issues
    
    def _analyze_algorithms(self, code: str) -> AlgorithmAnalysis:
        """Analyze algorithm efficiency"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return AlgorithmAnalysis("O(?)", "O(?)", 0, 0, {}, [])
        
        # Analyze time complexity
        time_complexity = self._estimate_time_complexity(tree)
        
        # Analyze space complexity
        space_complexity = self._estimate_space_complexity(tree)
        
        # Count loop nesting
        loop_nesting = self._calculate_loop_nesting(tree)
        
        # Count recursive calls
        recursive_calls = self._count_recursive_calls(tree)
        
        # Analyze data structure efficiency
        data_structure_efficiency = self._analyze_data_structures(tree)
        
        # Generate optimization suggestions
        optimizations = self._suggest_optimizations(tree)
        
        return AlgorithmAnalysis(
            estimated_time_complexity=time_complexity,
            estimated_space_complexity=space_complexity,
            loop_nesting_level=loop_nesting,
            recursive_calls=recursive_calls,
            data_structure_efficiency=data_structure_efficiency,
            potential_optimizations=optimizations
        )
    
    def _estimate_time_complexity(self, tree: ast.AST) -> str:
        """Estimate time complexity based on loop patterns"""
        nested_loops = 0
        max_nesting = 0
        current_nesting = 0
        
        def count_nested_loops(node, depth=0):
            nonlocal max_nesting, current_nesting
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.While)):
                    current_nesting = depth + 1
                    max_nesting = max(max_nesting, current_nesting)
                    count_nested_loops(child, depth + 1)
                else:
                    count_nested_loops(child, depth)
        
        count_nested_loops(tree)
        
        # Simple heuristic for time complexity
        if max_nesting == 0:
            return "O(1)"
        elif max_nesting == 1:
            return "O(n)"
        elif max_nesting == 2:
            return "O(n²)"
        elif max_nesting == 3:
            return "O(n³)"
        else:
            return f"O(n^{max_nesting})"
    
    def _estimate_space_complexity(self, tree: ast.AST) -> str:
        """Estimate space complexity based on data structure usage"""
        has_recursion = False
        has_data_structures = False
        
        for node in ast.walk(tree):
            # Check for recursive patterns
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                # Simple check - function calling itself by name
                has_recursion = True
            
            # Check for data structure creation
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['list', 'dict', 'set', 'tuple']:
                        has_data_structures = True
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['append', 'extend', 'update']:
                        has_data_structures = True
        
        if has_recursion:
            return "O(n)"  # Recursive call stack
        elif has_data_structures:
            return "O(n)"  # Additional data structures
        else:
            return "O(1)"  # Constant space
    
    def _calculate_loop_nesting(self, tree: ast.AST) -> int:
        """Calculate maximum loop nesting level"""
        max_nesting = 0
        
        def count_nesting(node, depth=0):
            nonlocal max_nesting
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.While)):
                    new_depth = depth + 1
                    max_nesting = max(max_nesting, new_depth)
                    count_nesting(child, new_depth)
                else:
                    count_nesting(child, depth)
        
        count_nesting(tree)
        return max_nesting
    
    def _count_recursive_calls(self, tree: ast.AST) -> int:
        """Count potential recursive function calls"""
        recursive_count = 0
        function_names = set()
        
        # Collect function names
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_names.add(node.name)
        
        # Count calls to those functions
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in function_names:
                    recursive_count += 1
        
        return recursive_count
    
    def _analyze_data_structures(self, tree: ast.AST) -> Dict[str, str]:
        """Analyze data structure usage efficiency"""
        analysis = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id == 'list':
                        analysis['list_usage'] = "O(1) append, O(n) search"
                    elif node.func.id == 'dict':
                        analysis['dict_usage'] = "O(1) average access"
                    elif node.func.id == 'set':
                        analysis['set_usage'] = "O(1) average membership"
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'sort':
                        analysis['sorting'] = "O(n log n) time complexity"
        
        return analysis
    
    def _suggest_optimizations(self, tree: ast.AST) -> List[str]:
        """Generate optimization suggestions"""
        suggestions = []
        
        # Check for inefficient patterns
        for node in ast.walk(tree):
            # Nested loops with list operations
            if isinstance(node, ast.For):
                for child in ast.walk(node):
                    if isinstance(child, ast.For):
                        suggestions.append("Consider using list comprehensions for nested loops")
                        break
            
            # String concatenation in loops
            if isinstance(node, ast.For):
                for child in ast.walk(node):
                    if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                        suggestions.append("Use join() instead of string concatenation in loops")
                        break
            
            # Linear search patterns
            if isinstance(node, ast.Compare):
                if any(isinstance(op, ast.In) for op in node.ops):
                    suggestions.append("Consider using sets or dicts for O(1) membership testing")
        
        return list(set(suggestions))  # Remove duplicates
    
    def _extract_algorithm_issues(self, analysis: AlgorithmAnalysis) -> List[VerificationIssue]:
        """Extract issues from algorithm analysis"""
        issues = []
        
        # Time complexity warnings
        if "³" in analysis.estimated_time_complexity or "^" in analysis.estimated_time_complexity:
            issues.append(VerificationIssue(
                type="time_complexity",
                severity=Severity.HIGH,
                message=f"High time complexity: {analysis.estimated_time_complexity}",
                suggestion="Consider algorithmic optimizations to reduce complexity"
            ))
        elif "²" in analysis.estimated_time_complexity:
            issues.append(VerificationIssue(
                type="time_complexity",
                severity=Severity.MEDIUM,
                message=f"Quadratic time complexity: {analysis.estimated_time_complexity}",
                suggestion="Monitor performance with large datasets"
            ))
        
        # Loop nesting warnings
        if analysis.loop_nesting_level >= 3:
            issues.append(VerificationIssue(
                type="loop_nesting",
                severity=Severity.HIGH,
                message=f"Deep loop nesting: {analysis.loop_nesting_level} levels",
                suggestion="Consider algorithmic improvements or caching"
            ))
        elif analysis.loop_nesting_level >= 2:
            issues.append(VerificationIssue(
                type="loop_nesting",
                severity=Severity.MEDIUM,
                message=f"Nested loops detected: {analysis.loop_nesting_level} levels",
                suggestion="Consider optimization for large inputs"
            ))
        
        # Add optimization suggestions as low-priority issues
        for optimization in analysis.potential_optimizations:
            issues.append(VerificationIssue(
                type="optimization",
                severity=Severity.LOW,
                message=f"Optimization opportunity: {optimization}",
                suggestion="Consider implementing this optimization for better performance"
            ))
        
        return issues
    
    def _static_performance_analysis(self, code: str) -> List[VerificationIssue]:
        """Perform static analysis for performance issues"""
        issues = []
        lines = code.splitlines()
        
        # Check for performance anti-patterns
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # String concatenation in loops (simplified detection)
            if re.search(r'for\s+\w+\s+in.*:', line_stripped):
                # Look ahead for string concatenation
                for next_line_num in range(line_num, min(line_num + 5, len(lines))):
                    if next_line_num < len(lines):
                        next_line = lines[next_line_num].strip()
                        if '+=' in next_line and ('"' in next_line or "'" in next_line):
                            issues.append(VerificationIssue(
                                type="performance_antipattern",
                                severity=Severity.MEDIUM,
                                message="String concatenation in loop detected",
                                line_number=next_line_num + 1,
                                suggestion="Use join() or f-strings for better performance"
                            ))
                            break
            
            # Global variable access patterns
            if re.search(r'global\s+\w+', line_stripped):
                issues.append(VerificationIssue(
                    type="performance_concern",
                    severity=Severity.LOW,
                    message="Global variable access can impact performance",
                    line_number=line_num,
                    suggestion="Consider passing variables as parameters"
                ))
            
            # Inefficient membership testing
            if re.search(r'\w+\s+in\s+\[.*\]', line_stripped):
                issues.append(VerificationIssue(
                    type="performance_antipattern",
                    severity=Severity.MEDIUM,
                    message="Inefficient membership testing with list",
                    line_number=line_num,
                    suggestion="Use sets for O(1) membership testing"
                ))
        
        return issues
    
    def _is_safe_for_execution(self, code: str) -> bool:
        """Check if code is safe for runtime profiling"""
        dangerous_patterns = [
            'import os', 'import sys', 'import subprocess', 'import shutil',
            'open(', 'file(', 'exec(', 'eval(', '__import__',
            'input(', 'raw_input(', 'while True:', 'infinite'
        ]
        
        return not any(pattern in code for pattern in dangerous_patterns)
    
    async def _runtime_profiling(self, code: str) -> PerformanceMetrics:
        """Perform runtime profiling of code"""
        if not PROFILING_AVAILABLE:
            return PerformanceMetrics(0.0, 0.0, 0.0, 0, 0, 1.0)
        
        # Create a temporary file for profiling
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Start memory tracking
            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            
            # Time execution
            start_time = time.time()
            
            # Execute with timeout
            process = await asyncio.create_subprocess_exec(
                sys.executable, temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.max_execution_time
                )
                
                execution_time = time.time() - start_time
                
                # Get memory usage
                current_memory, peak_memory = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                memory_used = (current_memory - start_memory) / 1024 / 1024  # MB
                peak_memory_mb = peak_memory / 1024 / 1024  # MB
                
                # Calculate performance score
                performance_score = self._calculate_runtime_score(
                    execution_time, memory_used, process.returncode == 0
                )
                
                return PerformanceMetrics(
                    execution_time=execution_time,
                    memory_usage=memory_used,
                    peak_memory=peak_memory_mb,
                    function_call_count=0,  # Would need more sophisticated profiling
                    io_operations=0,        # Would need more sophisticated profiling
                    performance_score=performance_score
                )
            
            except asyncio.TimeoutError:
                process.kill()
                tracemalloc.stop()
                return PerformanceMetrics(0.0, 0.0, 0.0, 0, 0, 0.0)
        
        finally:
            # Clean up
            try:
                os.unlink(temp_file)
            except Exception:
                pass
    
    def _calculate_runtime_score(self, execution_time: float, memory_usage: float, 
                                 success: bool) -> float:
        """Calculate performance score based on runtime metrics"""
        if not success:
            return 0.0
        
        # Score based on execution time (lower is better)
        time_score = max(0.0, 1.0 - (execution_time / self.max_execution_time))
        
        # Score based on memory usage (lower is better)
        memory_score = max(0.0, 1.0 - (memory_usage / self.memory_limit_mb))
        
        # Weighted average
        return (time_score * 0.6 + memory_score * 0.4)
    
    def _extract_runtime_issues(self, metrics: PerformanceMetrics) -> List[VerificationIssue]:
        """Extract issues from runtime performance metrics"""
        issues = []
        
        # Execution time issues
        if metrics.execution_time > self.max_execution_time * 0.8:
            issues.append(VerificationIssue(
                type="execution_time",
                severity=Severity.HIGH,
                message=f"Slow execution time: {metrics.execution_time:.3f}s",
                suggestion="Optimize algorithm or reduce computational complexity"
            ))
        elif metrics.execution_time > self.max_execution_time * 0.5:
            issues.append(VerificationIssue(
                type="execution_time",
                severity=Severity.MEDIUM,
                message=f"Moderate execution time: {metrics.execution_time:.3f}s",
                suggestion="Consider performance optimizations"
            ))
        
        # Memory usage issues
        if metrics.memory_usage > self.memory_limit_mb * 0.8:
            issues.append(VerificationIssue(
                type="memory_usage",
                severity=Severity.HIGH,
                message=f"High memory usage: {metrics.memory_usage:.2f} MB",
                suggestion="Optimize memory usage or use streaming approaches"
            ))
        elif metrics.memory_usage > self.memory_limit_mb * 0.5:
            issues.append(VerificationIssue(
                type="memory_usage",
                severity=Severity.MEDIUM,
                message=f"Moderate memory usage: {metrics.memory_usage:.2f} MB",
                suggestion="Monitor memory usage with larger datasets"
            ))
        
        return issues
    
    def _analyze_resource_usage(self, code: str) -> ResourceAnalysis:
        """Analyze resource usage patterns"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ResourceAnalysis(1.0, 1.0, 1.0, [], [])
        
        scalability_concerns = []
        bottleneck_locations = []
        
        # Analyze for scalability issues
        line_num = 1
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Nested loops are scalability concerns
                nested_loops = sum(1 for child in ast.walk(node) 
                                  if isinstance(child, (ast.For, ast.While)))
                if nested_loops > 1:
                    scalability_concerns.append(f"Nested loops at line {getattr(node, 'lineno', line_num)}")
                    bottleneck_locations.append((getattr(node, 'lineno', line_num), "Nested loops"))
            
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # File I/O operations
                    if node.func.attr in ['read', 'write', 'open']:
                        scalability_concerns.append(f"I/O operation at line {getattr(node, 'lineno', line_num)}")
        
        # Calculate efficiency scores (simplified)
        memory_efficiency = 1.0 - min(len(scalability_concerns) * 0.2, 0.8)
        cpu_efficiency = 1.0 - min(len(bottleneck_locations) * 0.3, 0.9)
        io_efficiency = 1.0  # Would need runtime analysis for accurate measurement
        
        return ResourceAnalysis(
            memory_efficiency=memory_efficiency,
            cpu_efficiency=cpu_efficiency,
            io_efficiency=io_efficiency,
            scalability_concerns=scalability_concerns,
            bottleneck_locations=bottleneck_locations
        )
    
    def _extract_resource_issues(self, analysis: ResourceAnalysis) -> List[VerificationIssue]:
        """Extract issues from resource analysis"""
        issues = []
        
        # Memory efficiency issues
        if analysis.memory_efficiency < 0.5:
            issues.append(VerificationIssue(
                type="memory_efficiency",
                severity=Severity.HIGH,
                message=f"Low memory efficiency: {analysis.memory_efficiency:.2f}",
                suggestion="Optimize data structures and memory usage patterns"
            ))
        elif analysis.memory_efficiency < 0.7:
            issues.append(VerificationIssue(
                type="memory_efficiency",
                severity=Severity.MEDIUM,
                message=f"Moderate memory efficiency: {analysis.memory_efficiency:.2f}",
                suggestion="Consider memory optimization opportunities"
            ))
        
        # CPU efficiency issues
        if analysis.cpu_efficiency < 0.5:
            issues.append(VerificationIssue(
                type="cpu_efficiency",
                severity=Severity.HIGH,
                message=f"Low CPU efficiency: {analysis.cpu_efficiency:.2f}",
                suggestion="Optimize algorithms and reduce computational complexity"
            ))
        elif analysis.cpu_efficiency < 0.7:
            issues.append(VerificationIssue(
                type="cpu_efficiency",
                severity=Severity.MEDIUM,
                message=f"Moderate CPU efficiency: {analysis.cpu_efficiency:.2f}",
                suggestion="Consider algorithmic improvements"
            ))
        
        # Scalability concerns
        for concern in analysis.scalability_concerns:
            issues.append(VerificationIssue(
                type="scalability",
                severity=Severity.MEDIUM,
                message=f"Scalability concern: {concern}",
                suggestion="Consider optimization for large-scale usage"
            ))
        
        # Bottleneck locations
        for line_num, description in analysis.bottleneck_locations:
            issues.append(VerificationIssue(
                type="bottleneck",
                severity=Severity.MEDIUM,
                message=f"Performance bottleneck: {description}",
                line_number=line_num,
                suggestion="Profile and optimize this section for better performance"
            ))
        
        return issues
    
    def _calculate_performance_score(self, issues: List[VerificationIssue], 
                                   metadata: Dict[str, Any]) -> float:
        """Calculate overall performance score with very aggressive penalty for complexity"""
        if not issues:
            return 1.0
        
        # Very aggressive weights for performance issues
        type_weights = {
            "complexity": 1.5,           # Very high penalty for cyclomatic complexity
            "cognitive_complexity": 2.0, # Cognitive complexity kills maintainability
            "nesting": 1.8,             # Deep nesting is extremely problematic
            "function_length": 0.8,     # Long functions hurt readability
            "parameter_count": 0.6,     # Too many params hurt usability
            "time_complexity": 1.5,     # Algorithm efficiency is critical
            "loop_nesting": 1.2,        # Nested loops hurt performance badly
            "performance_antipattern": 1.0,  # Anti-patterns matter
            "execution_time": 1.5,      # Actual slow execution is very bad
            "memory_usage": 1.2,        # Memory issues are serious
            "scalability": 0.8,         # Scalability concerns matter
            "bottleneck": 1.0,          # Bottlenecks hurt UX
            "optimization": 0.1         # Suggestions are just hints
        }
        
        # Very aggressive severity multipliers
        severity_multipliers = {
            Severity.LOW: 0.3,
            Severity.MEDIUM: 0.8,       # Medium issues get significant penalty
            Severity.HIGH: 1.5,         # High issues are major problems
            Severity.CRITICAL: 2.5      # Critical issues should destroy the score
        }
        
        total_penalty = 0.0
        for issue in issues:
            type_weight = type_weights.get(issue.type, 0.5)
            severity_multiplier = severity_multipliers[issue.severity]
            issue_penalty = type_weight * severity_multiplier
            total_penalty += issue_penalty
        
        # Very small runtime bonus (complexity issues should dominate)
        runtime_bonus = 0.0
        if 'performance_metrics' in metadata:
            perf_metrics = metadata['performance_metrics']
            if isinstance(perf_metrics, dict) and 'performance_score' in perf_metrics:
                runtime_bonus = perf_metrics['performance_score'] * 0.05  # Minimal bonus
        
        # Very aggressive normalization - even 2-3 critical issues should tank score
        # 2 critical issues should give score ~0.2, 1 critical + some others ~0.4
        max_penalty = 4.0  # Very aggressive - reduced from 6
        normalized_penalty = min(total_penalty / max_penalty, 1.0)
        
        base_score = max(0.0, 1.0 - normalized_penalty)
        return min(1.0, base_score + runtime_bonus)