# src/agents/performance_profiler.py
"""
Performance Profiler Agent - Scale-aware algorithmic intelligence with production deployment standards.
Implements context-aware complexity analysis and enterprise performance requirements.
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
    """Enhanced complexity metrics with scale awareness"""
    cyclomatic_complexity: int
    cognitive_complexity: int
    nesting_depth: int
    function_length: int
    parameter_count: int
    return_statements: int
    loop_count: int
    conditional_count: int
    algorithm_complexity_class: str  # O(1), O(n), O(n²), etc.
    scale_risk_factor: float  # 0.0-1.0 risk of performance issues at scale


@dataclass
class AlgorithmAnalysis:
    """Enhanced algorithm efficiency analysis with context intelligence"""
    estimated_time_complexity: str
    estimated_space_complexity: str
    loop_nesting_level: int
    recursive_calls: int
    algorithm_pattern: str  # "sorting", "searching", "traversal", etc.
    scale_suitability: str  # "small", "medium", "large", "enterprise"
    performance_bottlenecks: List[Tuple[int, str, float]]  # line, description, severity
    optimization_opportunities: List[str]


@dataclass
class PerformanceMetrics:
    """Enhanced runtime performance metrics"""
    execution_time: float
    memory_usage: float
    peak_memory: float
    function_call_count: int
    io_operations: int
    performance_score: float
    scalability_projection: Dict[str, float]  # projected performance at different scales


@dataclass
class ResourceAnalysis:
    """Enhanced resource usage analysis with production focus"""
    memory_efficiency: float
    cpu_efficiency: float
    io_efficiency: float
    scalability_concerns: List[str]
    bottleneck_locations: List[Tuple[int, str]]
    production_readiness: float  # 0.0-1.0 readiness for production deployment
    resource_leak_risk: float  # 0.0-1.0 risk of resource leaks


class PerformanceProfiler(BaseAgent):
    """
    Agent 3: Enterprise Performance Profiler
    
    Breakthrough features:
    - Scale-aware algorithmic intelligence with O(n) context awareness
    - Production deployment performance standards
    - Algorithm pattern recognition with enterprise suitability assessment
    - Memory leak detection and resource management validation
    - Context-aware performance penalty calculation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("PerformanceProfiler", config)
        self.max_execution_time = self.config.get('max_execution_time', 2.0)
        self.memory_limit_mb = self.config.get('memory_limit_mb', 100)
        self.enable_runtime_profiling = self.config.get('enable_runtime_profiling', True)
        
        # Enterprise performance thresholds
        self.enterprise_thresholds = {
            'cyclomatic_complexity': {'acceptable': 8, 'concerning': 12, 'unacceptable': 20},
            'cognitive_complexity': {'acceptable': 12, 'concerning': 20, 'unacceptable': 35},
            'nesting_depth': {'acceptable': 3, 'concerning': 5, 'unacceptable': 7},
            'function_length': {'acceptable': 30, 'concerning': 60, 'unacceptable': 100},
            'algorithm_complexity': {
                'O(1)': 'excellent',
                'O(log n)': 'excellent', 
                'O(n)': 'good',
                'O(n log n)': 'acceptable',
                'O(n²)': 'concerning',
                'O(n³)': 'unacceptable',
                'O(2^n)': 'critical'
            }
        }
        
        # Algorithm patterns for context-aware analysis
        self.algorithm_patterns = {
            'sorting': {
                'keywords': ['sort', 'sorted', 'order', 'arrange'],
                'optimal_complexity': 'O(n log n)',
                'scale_critical': True
            },
            'searching': {
                'keywords': ['find', 'search', 'locate', 'index'],
                'optimal_complexity': 'O(log n)',
                'scale_critical': True
            },
            'traversal': {
                'keywords': ['traverse', 'walk', 'iterate', 'visit'],
                'optimal_complexity': 'O(n)',
                'scale_critical': False
            },
            'filtering': {
                'keywords': ['filter', 'select', 'where', 'match'],
                'optimal_complexity': 'O(n)',
                'scale_critical': False
            }
        }
    
    async def _analyze_implementation(self, code: str, context: Dict[str, Any]) -> AgentResult:
        """Enhanced performance analysis with scale-aware intelligence"""
        issues = []
        metadata = {}
        
        # 1. Enhanced complexity analysis with scale awareness
        complexity_metrics = self._analyze_enhanced_complexity(code)
        complexity_issues = self._extract_scale_aware_complexity_issues(complexity_metrics, code)
        issues.extend(complexity_issues)
        metadata['complexity_metrics'] = complexity_metrics.__dict__
        
        # 2. Algorithm intelligence with pattern recognition
        algorithm_analysis = self._analyze_algorithm_intelligence(code)
        algorithm_issues = self._extract_algorithm_intelligence_issues(algorithm_analysis, code)
        issues.extend(algorithm_issues)
        metadata['algorithm_analysis'] = algorithm_analysis.__dict__
        
        # 3. Scale-aware performance analysis
        scale_issues = self._analyze_scale_performance(code)
        issues.extend(scale_issues)
        
        # 4. Memory leak and resource management analysis
        resource_analysis = self._analyze_resource_management(code)
        resource_issues = self._extract_resource_management_issues(resource_analysis)
        issues.extend(resource_issues)
        metadata['resource_analysis'] = resource_analysis.__dict__
        
        # 5. Production readiness assessment
        production_issues = self._assess_production_performance_readiness(code, issues)
        issues.extend(production_issues)
        
        # 6. Runtime profiling (if safe and enabled)
        if self.enable_runtime_profiling and self._is_safe_for_execution(code):
            try:
                performance_metrics = await self._enhanced_runtime_profiling(code)
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
        
        # Calculate enterprise performance score
        overall_score = self._calculate_enterprise_performance_score(issues, metadata)
        
        return AgentResult(
            agent_name=self.name,
            execution_time=0.0,
            overall_score=overall_score,
            issues=issues,
            metadata=metadata
        )
    
    def _analyze_enhanced_complexity(self, code: str) -> ComplexityMetrics:
        """Analyze complexity with scale-aware intelligence"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ComplexityMetrics(0, 0, 0, 0, 0, 0, 0, 0, "O(?)", 1.0)
        
        # Find all functions and analyze the most complex one
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        
        if not functions:
            return self._calculate_complexity_for_node(tree)
        
        # Analyze all functions and find the worst complexity
        max_complexity = ComplexityMetrics(0, 0, 0, 0, 0, 0, 0, 0, "O(1)", 0.0)
        
        for func in functions:
            complexity = self._calculate_complexity_for_node(func)
            if complexity.cyclomatic_complexity > max_complexity.cyclomatic_complexity:
                max_complexity = complexity
        
        return max_complexity
    
    def _calculate_complexity_for_node(self, node: ast.AST) -> ComplexityMetrics:
        """Calculate enhanced complexity metrics with algorithm classification"""
        cyclomatic = 1
        cognitive = 0
        nesting_depth = 0
        function_length = 0
        parameter_count = 0
        return_statements = 0
        loop_count = 0
        conditional_count = 0
        
        # Function-specific metrics
        if isinstance(node, ast.FunctionDef):
            parameter_count = len(node.args.args)
            if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                function_length = node.end_lineno - node.lineno
        
        # Enhanced complexity calculation
        nested_loops = 0
        max_loop_depth = 0
        current_loop_depth = 0
        
        for child in ast.walk(node):
            # Cyclomatic complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                cyclomatic += 1
                conditional_count += 1
                
                # Track nested loops specifically
                if isinstance(child, (ast.For, ast.While, ast.AsyncFor)):
                    loop_count += 1
                    
                    # Count nested loops within this loop
                    for nested_child in ast.walk(child):
                        if (isinstance(nested_child, (ast.For, ast.While, ast.AsyncFor)) and 
                            nested_child != child):
                            nested_loops += 1
            
            elif isinstance(child, ast.ExceptHandler):
                cyclomatic += 1
            elif isinstance(child, ast.With):
                cyclomatic += 1
            elif isinstance(child, ast.BoolOp):
                cyclomatic += len(child.values) - 1
            elif isinstance(child, ast.Return):
                return_statements += 1
        
        # Calculate nesting depth
        nesting_depth = self._calculate_nesting_depth(node)
        
        # Calculate cognitive complexity
        cognitive = self._calculate_cognitive_complexity(node)
        
        # Determine algorithm complexity class based on loop patterns
        algorithm_complexity_class = self._determine_algorithm_complexity(node, nested_loops, nesting_depth)
        
        # Calculate scale risk factor
        scale_risk_factor = self._calculate_scale_risk_factor(
            algorithm_complexity_class, nested_loops, nesting_depth, function_length
        )
        
        return ComplexityMetrics(
            cyclomatic_complexity=cyclomatic,
            cognitive_complexity=cognitive,
            nesting_depth=nesting_depth,
            function_length=function_length,
            parameter_count=parameter_count,
            return_statements=return_statements,
            loop_count=loop_count,
            conditional_count=conditional_count,
            algorithm_complexity_class=algorithm_complexity_class,
            scale_risk_factor=scale_risk_factor
        )
    
    def _determine_algorithm_complexity(self, node: ast.AST, nested_loops: int, max_loop_depth: int) -> str:
        """Determine algorithm complexity class based on structure"""
        if max_loop_depth == 0:
            return "O(1)"
        elif max_loop_depth == 1 and nested_loops == 0:
            return "O(n)"
        elif max_loop_depth >= 2 or nested_loops >= 1:
            return "O(n²)"
        elif max_loop_depth >= 3 or nested_loops >= 3:
            return "O(n³)"
        elif max_loop_depth >= 4:
            return "O(n^4+)"
        
        # Check for recursive patterns
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                # Simple recursion detection
                if hasattr(node, 'name') and child.func.id == node.name:
                    return "O(2^n)"  # Exponential for simple recursion
        
        return "O(n)"
    
    def _calculate_scale_risk_factor(self, complexity_class: str, nested_loops: int, 
                                   max_loop_depth: int, function_length: int) -> float:
        """Calculate risk factor for performance issues at scale"""
        base_risk = {
            "O(1)": 0.0,
            "O(log n)": 0.1,
            "O(n)": 0.2,
            "O(n log n)": 0.3,
            "O(n²)": 0.8,      # Very high risk for quadratic
            "O(n³)": 0.95,     # Critical risk for cubic
            "O(n^4+)": 1.0,    # Unacceptable
            "O(2^n)": 1.0      # Unacceptable
        }.get(complexity_class, 0.5)
        
        # Increase risk based on nested loops
        loop_risk = min(0.3, nested_loops * 0.15)  # More aggressive
        
        # Increase risk based on function length
        length_risk = min(0.2, max(0, (function_length - 30) / 70))  # More aggressive threshold
        
        # Increase risk based on deep nesting
        nesting_risk = min(0.2, max(0, (max_loop_depth - 1) * 0.15))  # More aggressive
        
        return min(1.0, base_risk + loop_risk + length_risk + nesting_risk)
    
    def _calculate_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth"""
        max_depth = current_depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                child_depth = self._calculate_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _calculate_cognitive_complexity(self, node: ast.AST, nesting_level: int = 0) -> int:
        """Calculate cognitive complexity"""
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
    
    def _extract_scale_aware_complexity_issues(self, metrics: ComplexityMetrics, code: str) -> List[VerificationIssue]:
        """Extract complexity issues with scale-aware severity"""
        issues = []
        
        # Check if this is simple code that shouldn't have critical performance issues
        lines = code.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        is_simple_code = len(non_empty_lines) < 30
        
        # Don't flag simple code with basic algorithms as critical
        complexity_assessment = self.enterprise_thresholds['algorithm_complexity'].get(
            metrics.algorithm_complexity_class, 'unknown'
        )
        
        if complexity_assessment == 'critical':
            # Only flag as critical if it's actually complex code
            severity = Severity.HIGH if is_simple_code else Severity.CRITICAL
            issues.append(VerificationIssue(
                type="algorithm_complexity",
                severity=severity,
                message=f"Algorithm complexity: {metrics.algorithm_complexity_class}",
                suggestion="Consider algorithm optimization" if is_simple_code else "Redesign algorithm with better complexity class"
            ))
        elif complexity_assessment == 'concerning' and not is_simple_code:
            # Only flag concerning complexity in non-simple code
            issues.append(VerificationIssue(
                type="algorithm_complexity",
                severity=Severity.MEDIUM,
                message=f"Concerning algorithm complexity: {metrics.algorithm_complexity_class}",
                suggestion="Consider algorithmic optimization for production scale"
            ))
        
        # Less harsh complexity penalties for simple code
        thresholds = self.enterprise_thresholds['cyclomatic_complexity']
        if metrics.cyclomatic_complexity >= thresholds['unacceptable']:
            severity = Severity.HIGH if is_simple_code else Severity.CRITICAL
            issues.append(VerificationIssue(
                type="cyclomatic_complexity",
                severity=severity,
                message=f"High cyclomatic complexity: {metrics.cyclomatic_complexity}",
                suggestion="Consider refactoring" if is_simple_code else "Break function into smaller components"
            ))
        
        return issues
   
    def _is_scale_critical_pattern(self, code: str) -> bool:
        """Check if code contains scale-critical algorithm patterns"""
        code_lower = code.lower()
        
        for pattern_name, pattern_info in self.algorithm_patterns.items():
            if pattern_info['scale_critical']:
                for keyword in pattern_info['keywords']:
                    if keyword in code_lower:
                        return True
        
        return False
    
    def _analyze_algorithm_intelligence(self, code: str) -> AlgorithmAnalysis:
        """Analyze algorithm intelligence with pattern recognition"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return AlgorithmAnalysis("O(?)", "O(?)", 0, 0, "unknown", "small", [], [])
        
        # Analyze time complexity
        time_complexity = self._estimate_time_complexity(tree)
        
        # Analyze space complexity  
        space_complexity = self._estimate_space_complexity(tree)
        
        # Count loop nesting
        loop_nesting = self._calculate_loop_nesting_level(tree)
        
        # Count recursive calls
        recursive_calls = self._count_recursive_calls(tree)
        
        # Identify algorithm pattern
        algorithm_pattern = self._identify_algorithm_pattern(code, tree)
        
        # Assess scale suitability
        scale_suitability = self._assess_scale_suitability(time_complexity, space_complexity, algorithm_pattern)
        
        # Find performance bottlenecks
        bottlenecks = self._find_performance_bottlenecks(tree, code)
        
        # Generate optimization suggestions
        optimizations = self._generate_optimization_suggestions(tree, code, algorithm_pattern)
        
        return AlgorithmAnalysis(
            estimated_time_complexity=time_complexity,
            estimated_space_complexity=space_complexity,
            loop_nesting_level=loop_nesting,
            recursive_calls=recursive_calls,
            algorithm_pattern=algorithm_pattern,
            scale_suitability=scale_suitability,
            performance_bottlenecks=bottlenecks,
            optimization_opportunities=optimizations
        )
    
    def _estimate_time_complexity(self, tree: ast.AST) -> str:
        """Enhanced time complexity estimation"""
        max_nesting = 0
        has_recursion = False
        nested_collections = 0
        
        def analyze_node(node, depth=0):
            nonlocal max_nesting, has_recursion, nested_collections
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.While)):
                    current_depth = depth + 1
                    max_nesting = max(max_nesting, current_depth)
                    
                    # Check for nested collection operations
                    for nested_child in ast.walk(child):
                        if isinstance(nested_child, ast.Call) and isinstance(nested_child.func, ast.Attribute):
                            if nested_child.func.attr in ['append', 'extend', 'insert', 'remove']:
                                nested_collections += 1
                    
                    analyze_node(child, current_depth)
                else:
                    analyze_node(child, depth)
        
        # Check for recursion
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                has_recursion = True  # Simplified check
        
        analyze_node(tree)
        
        # Enhanced complexity determination
        if has_recursion and max_nesting == 0:
            return "O(2^n)"  # Exponential recursion
        elif max_nesting == 0:
            return "O(1)"
        elif max_nesting == 1 and nested_collections == 0:
            return "O(n)"
        elif max_nesting == 1 and nested_collections > 0:
            return "O(n²)"  # Linear loop with quadratic operations
        elif max_nesting == 2:
            return "O(n²)"
        elif max_nesting == 3:
            return "O(n³)"
        else:
            return f"O(n^{max_nesting})"
    
    def _estimate_space_complexity(self, tree: ast.AST) -> str:
        """Enhanced space complexity estimation"""
        has_recursion = False
        creates_collections = False
        collection_operations = 0
        
        for node in ast.walk(tree):
            # Check for recursive patterns
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                has_recursion = True
            
            # Check for collection creation
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['list', 'dict', 'set', 'tuple']:
                        creates_collections = True
                        collection_operations += 1
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['append', 'extend', 'copy']:
                        creates_collections = True
                        collection_operations += 1
        
        if has_recursion:
            return "O(n)"  # Recursive call stack
        elif collection_operations > 2:
            return "O(n)"  # Multiple data structures
        elif creates_collections:
            return "O(n)"  # Additional data structures
        else:
            return "O(1)"  # Constant space
    
    def _calculate_loop_nesting_level(self, tree: ast.AST) -> int:
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
    
    def _identify_algorithm_pattern(self, code: str, tree: ast.AST) -> str:
        """Identify algorithm pattern from code"""
        code_lower = code.lower()
        
        # Check for algorithm patterns
        for pattern_name, pattern_info in self.algorithm_patterns.items():
            for keyword in pattern_info['keywords']:
                if keyword in code_lower:
                    return pattern_name
        
        # Analyze AST structure for patterns
        has_loops = any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
        has_conditions = any(isinstance(node, ast.If) for node in ast.walk(tree))
        has_collections = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) 
                                and node.func.id in ['list', 'dict', 'set'] 
                                for node in ast.walk(tree))
        
        if has_loops and has_conditions:
            return "conditional_traversal"
        elif has_loops:
            return "traversal"
        elif has_conditions:
            return "filtering"
        elif has_collections:
            return "data_processing"
        else:
            return "unknown"
    
    def _assess_scale_suitability(self, time_complexity: str, space_complexity: str, pattern: str) -> str:
        """Assess algorithm suitability for different scales"""
        complexity_scale_map = {
            "O(1)": "enterprise",
            "O(log n)": "enterprise", 
            "O(n)": "large",
            "O(n log n)": "large",
            "O(n²)": "medium",
            "O(n³)": "small",
            "O(2^n)": "tiny"
        }
        
        time_scale = complexity_scale_map.get(time_complexity, "small")
        
        # Adjust based on algorithm pattern criticality
        if pattern in ['sorting', 'searching'] and time_complexity in ["O(n²)", "O(n³)"]:
            return "tiny"  # Scale-critical patterns need better complexity
        
        return time_scale
    
    def _find_performance_bottlenecks(self, tree: ast.AST, code: str) -> List[Tuple[int, str, float]]:
        """Find specific performance bottlenecks in code"""
        bottlenecks = []
        lines = code.splitlines()
        
        for node in ast.walk(tree):
            line_num = getattr(node, 'lineno', 0)
            
            # Nested loops bottleneck
            if isinstance(node, (ast.For, ast.While)):
                nested_count = sum(1 for child in ast.walk(node) 
                                    if isinstance(child, (ast.For, ast.While)) and child != node)
                if nested_count >= 1:
                    severity = min(1.0, nested_count * 0.4)
                    bottlenecks.append((line_num, f"Nested loops ({nested_count+1} levels)", severity))
            
            # String concatenation in loops
            elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
                # Check if we're in a loop
                for parent in ast.walk(tree):
                    if isinstance(parent, (ast.For, ast.While)):
                        for child in ast.walk(parent):
                            if child == node:
                                bottlenecks.append((line_num, "String concatenation in loop", 0.6))
                                break
            
            # Inefficient membership testing
            elif isinstance(node, ast.Compare):
                for op in node.ops:
                    if isinstance(op, ast.In):
                        # Check if comparing against list literal
                        for comparator in node.comparators:
                            if isinstance(comparator, (ast.List, ast.Tuple)):
                                bottlenecks.append((line_num, "Linear search in list/tuple", 0.4))
        
        return bottlenecks
    
    def _generate_optimization_suggestions(self, tree: ast.AST, code: str, pattern: str) -> List[str]:
        """Generate context-aware optimization suggestions"""
        suggestions = []
        
        # Pattern-specific suggestions
        if pattern == "sorting":
            if "sort" in code.lower() or "sorted" in code.lower():
                suggestions.append("Use built-in sort() or sorted() for optimal O(n log n) performance")
        elif pattern == "searching":
            if any(op in code.lower() for op in ["find", "search", "index"]):
                suggestions.append("Consider using sets or dicts for O(1) lookup performance")
        
        # General algorithmic suggestions
        nested_loops = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if isinstance(child, (ast.For, ast.While)) and child != node:
                        nested_loops += 1
        
        if nested_loops > 0:
            suggestions.append("Replace nested loops with more efficient algorithms or data structures")
        
        # String operation suggestions
        if re.search(r'for.*\+.*str', code):
            suggestions.append("Use join() for string concatenation instead of repeated concatenation")
        
        # Collection operation suggestions  
        if re.search(r'\.append.*for.*in', code):
            suggestions.append("Consider list comprehensions for better performance")
        
        return suggestions
    
    def _extract_algorithm_intelligence_issues(self, analysis: AlgorithmAnalysis, code: str) -> List[VerificationIssue]:
        """Extract issues from algorithm intelligence analysis"""
        issues = []
        
        # Scale suitability issues
        if analysis.scale_suitability == "tiny":
            issues.append(VerificationIssue(
                type="scale_unsuitability",
                severity=Severity.CRITICAL,
                message=f"Algorithm unsuitable for production scale - {analysis.estimated_time_complexity} complexity",
                suggestion="Redesign with more efficient algorithm for enterprise deployment"
            ))
        elif analysis.scale_suitability == "small":
            if analysis.algorithm_pattern in ['sorting', 'searching']:
                issues.append(VerificationIssue(
                    type="scale_unsuitability",
                    severity=Severity.CRITICAL,
                    message=f"Scale-critical {analysis.algorithm_pattern} algorithm with poor complexity",
                    suggestion="Use optimized algorithm for scale-critical operations"
                ))
            else:
                issues.append(VerificationIssue(
                    type="scale_unsuitability",
                    severity=Severity.HIGH,
                    message=f"Algorithm only suitable for small datasets - {analysis.estimated_time_complexity}",
                    suggestion="Optimize algorithm for production data volumes"
                ))
        elif analysis.scale_suitability == "medium":
            issues.append(VerificationIssue(
                type="scale_concern",
                severity=Severity.MEDIUM,
                message=f"Algorithm may have performance issues at enterprise scale",
                suggestion="Monitor performance with production data volumes"
            ))
        
        # Loop nesting issues
        if analysis.loop_nesting_level >= 3:
            issues.append(VerificationIssue(
                type="loop_nesting",
                severity=Severity.CRITICAL,
                message=f"Excessive loop nesting: {analysis.loop_nesting_level} levels - performance bomb",
                suggestion="Redesign algorithm to eliminate deep nesting"
            ))
        elif analysis.loop_nesting_level >= 2:
            issues.append(VerificationIssue(
                type="loop_nesting",
                severity=Severity.HIGH,
                message=f"High loop nesting: {analysis.loop_nesting_level} levels",
                suggestion="Consider algorithmic optimization to reduce nesting"
            ))
        
        # Performance bottleneck issues
        for line_num, description, severity in analysis.performance_bottlenecks:
            issue_severity = Severity.CRITICAL if severity >= 0.8 else Severity.HIGH if severity >= 0.5 else Severity.MEDIUM
            issues.append(VerificationIssue(
                type="performance_bottleneck",
                severity=issue_severity,
                message=f"Performance bottleneck: {description}",
                line_number=line_num,
                suggestion="Optimize this performance-critical section"
            ))
        
        # Add optimization opportunities as suggestions
        for optimization in analysis.optimization_opportunities:
            issues.append(VerificationIssue(
                type="optimization_opportunity",
                severity=Severity.LOW,
                message=f"Optimization opportunity: {optimization}",
                suggestion="Consider implementing this optimization"
            ))
        
        return issues
    
    def _analyze_scale_performance(self, code: str) -> List[VerificationIssue]:
        """Analyze scale-specific performance issues"""
        issues = []
        lines = code.splitlines()
        
        # Scale-aware pattern detection
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Bubble sort detection (O(n²) sorting)
            if re.search(r'for.*for.*if.*>.*:', line_stripped):
                next_lines = lines[line_num:line_num+3] if line_num < len(lines)-2 else []
                if any('swap' in next_line or '=' in next_line for next_line in next_lines):
                    issues.append(VerificationIssue(
                        type="algorithm_inefficiency",
                        severity=Severity.CRITICAL,
                        message="Bubble sort detected - O(n²) sorting algorithm unacceptable for production",
                        line_number=line_num,
                        suggestion="Use built-in sorted() or list.sort() for O(n log n) performance"
                    ))
            
            # Linear search in sorted data
            if 'for' in line_stripped and any(word in line_stripped for word in ['find', 'search']):
                if 'sorted' in ''.join(lines[max(0, line_num-5):line_num]):
                    issues.append(VerificationIssue(
                        type="algorithm_inefficiency",
                        severity=Severity.HIGH,
                        message="Linear search in sorted data - use binary search for O(log n)",
                        line_number=line_num,
                        suggestion="Use bisect module for binary search in sorted data"
                    ))
            
            # String concatenation patterns
            if '+=' in line_stripped and ('"' in line_stripped or "'" in line_stripped):
                # Check if in loop
                indent = len(line) - len(line.lstrip())
                for check_line in lines[max(0, line_num-10):line_num]:
                    check_indent = len(check_line) - len(check_line.lstrip())
                    if check_indent < indent and 'for' in check_line:
                        issues.append(VerificationIssue(
                            type="performance_antipattern",
                            severity=Severity.HIGH,
                            message="String concatenation in loop - O(n²) performance degradation",
                            line_number=line_num,
                            suggestion="Use join() or f-strings for O(n) string building"
                        ))
                        break
        
        return issues
    
    def _analyze_resource_management(self, code: str) -> ResourceAnalysis:
        """Analyze resource management with production focus"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ResourceAnalysis(1.0, 1.0, 1.0, [], [], 0.5, 1.0)
        
        scalability_concerns = []
        bottleneck_locations = []
        resource_leak_risks = []
        
        # Analyze for resource leaks and scalability issues
        for node in ast.walk(tree):
            line_num = getattr(node, 'lineno', 1)
            
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                # File operations without context managers
                if node.func.id in ['open', 'file'] and not self._is_in_with_statement(node, tree):
                    resource_leak_risks.append(f"File operation without 'with' statement at line {line_num}")
                    scalability_concerns.append(f"Resource leak risk at line {line_num}")
                
                # Memory-intensive operations
                elif node.func.id in ['list', 'dict', 'set'] and self._in_loop_context(node, tree):
                    scalability_concerns.append(f"Collection creation in loop at line {line_num}")
                    bottleneck_locations.append((line_num, "Memory allocation in loop"))
            
            elif isinstance(node, ast.For):
                # Nested loops are scalability concerns
                nested_loops = sum(1 for child in ast.walk(node) 
                                    if isinstance(child, (ast.For, ast.While)))
                if nested_loops > 1:
                    scalability_concerns.append(f"Nested loops creating O(n²+) complexity at line {line_num}")
                    bottleneck_locations.append((line_num, f"Nested loops ({nested_loops} levels)"))
        
        # Calculate efficiency scores
        memory_efficiency = max(0.0, 1.0 - len(resource_leak_risks) * 0.3)
        cpu_efficiency = max(0.0, 1.0 - len(bottleneck_locations) * 0.2)
        io_efficiency = 1.0  # Would need runtime analysis
        
        # Calculate production readiness
        production_readiness = (memory_efficiency + cpu_efficiency + io_efficiency) / 3
        
        # Calculate resource leak risk
        resource_leak_risk = min(1.0, len(resource_leak_risks) * 0.4)
        
        return ResourceAnalysis(
            memory_efficiency=memory_efficiency,
            cpu_efficiency=cpu_efficiency,
            io_efficiency=io_efficiency,
            scalability_concerns=scalability_concerns,
            bottleneck_locations=bottleneck_locations,
            production_readiness=production_readiness,
            resource_leak_risk=resource_leak_risk
        )
    
    def _is_in_with_statement(self, target_node: ast.AST, tree: ast.AST) -> bool:
        """Check if node is within a with statement"""
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                for child in ast.walk(node):
                    if child is target_node:
                        return True
        return False
    
    def _in_loop_context(self, target_node: ast.AST, tree: ast.AST) -> bool:
        """Check if node is within a loop"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if child is target_node:
                        return True
        return False
    
    def _extract_resource_management_issues(self, analysis: ResourceAnalysis) -> List[VerificationIssue]:
        """Extract resource management issues"""
        issues = []
        
        # Memory efficiency issues
        if analysis.memory_efficiency < 0.5:
            issues.append(VerificationIssue(
                type="memory_efficiency",
                severity=Severity.CRITICAL,
                message=f"Critical memory efficiency: {analysis.memory_efficiency:.2f}",
                suggestion="Fix memory leaks and optimize memory usage for production"
            ))
        elif analysis.memory_efficiency < 0.7:
            issues.append(VerificationIssue(
                type="memory_efficiency",
                severity=Severity.HIGH,
                message=f"Poor memory efficiency: {analysis.memory_efficiency:.2f}",
                suggestion="Optimize memory usage patterns for production deployment"
            ))
        
        # CPU efficiency issues
        if analysis.cpu_efficiency < 0.5:
            issues.append(VerificationIssue(
                type="cpu_efficiency",
                severity=Severity.CRITICAL,
                message=f"Critical CPU efficiency: {analysis.cpu_efficiency:.2f}",
                suggestion="Optimize algorithms for production CPU performance"
            ))
        elif analysis.cpu_efficiency < 0.7:
            issues.append(VerificationIssue(
                type="cpu_efficiency",
                severity=Severity.HIGH,
                message=f"Poor CPU efficiency: {analysis.cpu_efficiency:.2f}",
                suggestion="Consider algorithmic improvements for production performance"
            ))
        
        # Resource leak risk
        if analysis.resource_leak_risk > 0.6:
            issues.append(VerificationIssue(
                type="resource_leak_risk",
                severity=Severity.CRITICAL,
                message=f"High resource leak risk: {analysis.resource_leak_risk:.2f}",
                suggestion="Fix resource management to prevent production memory leaks"
            ))
        elif analysis.resource_leak_risk > 0.3:
            issues.append(VerificationIssue(
                type="resource_leak_risk",
                severity=Severity.HIGH,
                message=f"Moderate resource leak risk: {analysis.resource_leak_risk:.2f}",
                suggestion="Improve resource cleanup for production stability"
            ))
        
        # Production readiness
        if analysis.production_readiness < 0.6:
            issues.append(VerificationIssue(
                type="production_readiness",
                severity=Severity.HIGH,
                message=f"Low production readiness: {analysis.production_readiness:.1%}",
                suggestion="Address performance issues before production deployment"
            ))
        
        # Scalability concerns
        for concern in analysis.scalability_concerns:
            issues.append(VerificationIssue(
                type="scalability_concern",
                severity=Severity.MEDIUM,
                message=f"Scalability concern: {concern}",
                suggestion="Optimize for production scale requirements"
            ))
        
        # Bottleneck locations
        for line_num, description in analysis.bottleneck_locations:
            issues.append(VerificationIssue(
                type="performance_bottleneck",
                severity=Severity.HIGH,
                message=f"Performance bottleneck: {description}",
                line_number=line_num,
                suggestion="Optimize this performance-critical section"
            ))
        
        return issues
    
    def _assess_production_performance_readiness(self, code: str, existing_issues: List[VerificationIssue]) -> List[VerificationIssue]:
        """Assess overall production performance readiness"""
        issues = []
        
        # Count critical performance issues
        critical_perf_issues = len([i for i in existing_issues 
                                    if i.severity == Severity.CRITICAL and 
                                    i.type in ['algorithm_complexity', 'scale_unsuitability', 'cpu_efficiency']])
        
        high_perf_issues = len([i for i in existing_issues 
                                if i.severity == Severity.HIGH and
                                i.type in ['algorithm_complexity', 'performance_bottleneck', 'memory_efficiency']])
        
        # Production deployment assessment
        if critical_perf_issues > 0:
            issues.append(VerificationIssue(
                type="production_performance_blocker",
                severity=Severity.CRITICAL,
                message=f"Production deployment blocked: {critical_perf_issues} critical performance issues",
                suggestion="Resolve all critical performance issues before production deployment"
            ))
        elif high_perf_issues >= 3:
            issues.append(VerificationIssue(
                type="production_performance_risk",
                severity=Severity.HIGH,
                message=f"High production performance risk: {high_perf_issues} high-severity issues",
                suggestion="Address performance issues to ensure production stability"
            ))
        elif high_perf_issues >= 1:
            issues.append(VerificationIssue(
                type="production_performance_concern",
                severity=Severity.MEDIUM,
                message=f"Production performance concerns: {high_perf_issues} issues need attention",
                suggestion="Monitor performance closely in production environment"
            ))
        
        return issues
    
    def _is_safe_for_execution(self, code: str) -> bool:
        """Check if code is safe for runtime profiling"""
        dangerous_patterns = [
            'import os', 'import sys', 'import subprocess', 'import shutil',
            'open(', 'file(', 'exec(', 'eval(', '__import__',
            'input(', 'raw_input(', 'while True:', 'infinite', 'socket'
        ]
        
        return not any(pattern in code for pattern in dangerous_patterns)
    
    async def _enhanced_runtime_profiling(self, code: str) -> PerformanceMetrics:
        """Enhanced runtime profiling with scalability projections"""
        if not PROFILING_AVAILABLE:
            return PerformanceMetrics(0.0, 0.0, 0.0, 0, 0, 1.0, {})
        
        # Create temporary file for profiling
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
                
                # Project scalability
                scalability_projection = self._project_scalability(execution_time, memory_used)
                
                return PerformanceMetrics(
                    execution_time=execution_time,
                    memory_usage=memory_used,
                    peak_memory=peak_memory_mb,
                    function_call_count=0,  # Would need instrumentation
                    io_operations=0,        # Would need instrumentation
                    performance_score=performance_score,
                    scalability_projection=scalability_projection
                )
            
            except asyncio.TimeoutError:
                process.kill()
                tracemalloc.stop()
                return PerformanceMetrics(0.0, 0.0, 0.0, 0, 0, 0.0, {})
        
        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass
    
    def _project_scalability(self, execution_time: float, memory_usage: float) -> Dict[str, float]:
        """Project performance at different scales"""
        return {
            '10x_data': execution_time * 10,
            '100x_data': execution_time * 100,
            '1000x_data': execution_time * 1000,
            'memory_10x': memory_usage * 10,
            'memory_100x': memory_usage * 100
        }
    
    def _calculate_runtime_score(self, execution_time: float, memory_usage: float, 
                                    success: bool) -> float:
        """Calculate performance score based on runtime metrics"""
        if not success:
            return 0.0
        
        # More aggressive scoring for production requirements
        time_score = max(0.0, 1.0 - (execution_time / (self.max_execution_time * 0.5)))  # Expect 50% of max time
        memory_score = max(0.0, 1.0 - (memory_usage / (self.memory_limit_mb * 0.5)))    # Expect 50% of max memory
        
        # Weighted average favoring execution time
        return (time_score * 0.7 + memory_score * 0.3)
    
    def _extract_runtime_issues(self, metrics: PerformanceMetrics) -> List[VerificationIssue]:
        """Extract issues from runtime performance metrics"""
        issues = []
        
        # Execution time issues (more aggressive)
        if metrics.execution_time > self.max_execution_time * 0.6:
            issues.append(VerificationIssue(
                type="execution_time",
                severity=Severity.CRITICAL,
                message=f"Slow execution time: {metrics.execution_time:.3f}s - production performance blocker",
                suggestion="Optimize algorithm for production performance requirements"
            ))
        elif metrics.execution_time > self.max_execution_time * 0.3:
            issues.append(VerificationIssue(
                type="execution_time",
                severity=Severity.HIGH,
                message=f"Concerning execution time: {metrics.execution_time:.3f}s",
                suggestion="Optimize algorithm for better production performance"
            ))
        
        # Memory usage issues (more aggressive)
        if metrics.memory_usage > self.memory_limit_mb * 0.6:
            issues.append(VerificationIssue(
                type="memory_usage",
                severity=Severity.CRITICAL,
                message=f"High memory usage: {metrics.memory_usage:.2f} MB - production deployment risk",
                suggestion="Optimize memory usage for production deployment"
            ))
        elif metrics.memory_usage > self.memory_limit_mb * 0.3:
            issues.append(VerificationIssue(
                type="memory_usage",
                severity=Severity.HIGH,
                message=f"Elevated memory usage: {metrics.memory_usage:.2f} MB",
                suggestion="Consider memory optimization for production scale"
            ))
        
        # Scalability projection issues
        if 'scalability_projection' in metrics.__dict__:
            proj = metrics.scalability_projection
            if '100x_data' in proj and proj['100x_data'] > 60:  # 1 minute for 100x data
                issues.append(VerificationIssue(
                    type="scalability_projection",
                    severity=Severity.CRITICAL,
                    message=f"Poor scalability projection: {proj['100x_data']:.1f}s for 100x data",
                    suggestion="Algorithm will not scale to production data volumes"
                ))
        
        return issues
    
    def _calculate_enterprise_performance_score(self, issues: List[VerificationIssue], 
                                            metadata: Dict[str, Any]) -> float:
        """Calculate enterprise performance score with less aggressive penalties for good code"""
        if not issues:
            return 1.0
        
        # Less aggressive weights for performance issues
        type_weights = {
            "algorithm_complexity": 1.5,        # Reduced from 2.0
            "scale_unsuitability": 1.8,         # Reduced from 2.5
            "cyclomatic_complexity": 0.8,       # Reduced from 1.2
            "cognitive_complexity": 0.6,        # Reduced from 1.0
            "nesting_depth": 1.0,               # Reduced from 1.5
            "function_length": 0.4,             # Reduced from 0.8
            "loop_nesting": 1.5,                # Reduced from 2.0
            "performance_bottleneck": 1.2,      # Reduced from 1.8
            "algorithm_inefficiency": 1.5,      # Reduced from 2.2
            "performance_antipattern": 1.0,     # Reduced from 1.5
            "memory_efficiency": 1.0,           # Reduced from 1.5
            "cpu_efficiency": 1.2,              # Reduced from 1.8
            "resource_leak_risk": 1.5,          # Reduced from 2.0
            "scalability_concern": 0.6,         # Reduced from 1.0
            "production_readiness": 1.0,        # Reduced from 1.5
            "execution_time": 1.0,              # Reduced from 1.5
            "memory_usage": 0.8,                # Reduced from 1.2
            "scalability_projection": 1.2,      # Reduced from 1.8
            "optimization_opportunity": 0.05    # Reduced from 0.1
        }
        
        # Less aggressive severity multipliers
        severity_multipliers = {
            Severity.LOW: 0.1,
            Severity.MEDIUM: 0.3,               # Reduced from 0.6
            Severity.HIGH: 0.8,                 # Reduced from 1.4
            Severity.CRITICAL: 1.5              # Reduced from 2.5
        }
        
        total_penalty = 0.0
        for issue in issues:
            type_weight = type_weights.get(issue.type, 0.5)
            severity_multiplier = severity_multipliers[issue.severity]
            issue_penalty = type_weight * severity_multiplier
            total_penalty += issue_penalty
        
        # Bonus for good runtime performance
        runtime_bonus = 0.0
        if 'performance_metrics' in metadata:
            perf_metrics = metadata['performance_metrics']
            if isinstance(perf_metrics, dict):
                perf_score = perf_metrics.get('performance_score', 0)
                if perf_score > 0.9:
                    runtime_bonus = 0.1  # Small bonus for excellent runtime performance
        
        # Less aggressive normalization - higher max_penalty means gentler scoring
        max_penalty = 5.0  # Increased from 3.0 for less aggressive scoring
        normalized_penalty = min(total_penalty / max_penalty, 1.0)
        
        base_score = max(0.2, 1.0 - normalized_penalty)  # Higher floor from 0.0
        final_score = min(1.0, base_score + runtime_bonus)
        
        return final_score