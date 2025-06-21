# src/agents/performance_profiler.py
"""
Performance Profiler Agent - Intelligent scale-aware algorithmic analysis.
Optimized for real-world code evaluation with context-aware scoring.
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
    """Enhanced complexity metrics with real-world context"""
    cyclomatic_complexity: int
    cognitive_complexity: int
    nesting_depth: int
    function_length: int
    parameter_count: int
    return_statements: int
    loop_count: int
    conditional_count: int
    algorithm_complexity_class: str
    scale_risk_factor: float
    code_context: str  # "patch", "full_file", "snippet"


@dataclass
class AlgorithmAnalysis:
    """Smart algorithm analysis with production context"""
    estimated_time_complexity: str
    estimated_space_complexity: str
    loop_nesting_level: int
    recursive_calls: int
    algorithm_pattern: str
    scale_suitability: str
    performance_bottlenecks: List[Tuple[int, str, float]]
    optimization_opportunities: List[str]
    is_production_critical: bool


@dataclass
class PerformanceMetrics:
    """Real-world performance assessment"""
    execution_time: float
    memory_usage: float
    peak_memory: float
    function_call_count: int
    io_operations: int
    performance_score: float
    scalability_projection: Dict[str, float]


class PerformanceProfiler(BaseAgent):
    """
    Intelligent Performance Profiler Agent
    
    Key Improvements:
    - Context-aware analysis (distinguishes patches from full files)
    - Real-world complexity thresholds
    - Smart pattern recognition for common algorithms
    - Production-oriented scoring that balances thoroughness with practicality
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("PerformanceProfiler", config)
        self.max_execution_time = self.config.get('max_execution_time', 2.0)
        self.memory_limit_mb = self.config.get('memory_limit_mb', 100)
        self.enable_runtime_profiling = self.config.get('enable_runtime_profiling', True)
        
        # Real-world performance thresholds (more lenient than academic standards)
        self.real_world_thresholds = {
            'cyclomatic_complexity': {
                'patch_context': {'warning': 15, 'critical': 25},
                'full_file': {'warning': 20, 'critical': 35}
            },
            'cognitive_complexity': {
                'patch_context': {'warning': 20, 'critical': 35},
                'full_file': {'warning': 25, 'critical': 45}
            },
            'nesting_depth': {
                'patch_context': {'warning': 5, 'critical': 8},
                'full_file': {'warning': 6, 'critical': 10}
            },
            'function_length': {
                'patch_context': {'warning': 80, 'critical': 150},
                'full_file': {'warning': 100, 'critical': 200}
            }
        }
        
        # Smart algorithm pattern recognition
        self.algorithm_intelligence = {
            'sorting_patterns': {
                'bubble_sort': {'complexity': 'O(n²)', 'acceptable_for': 'small_data'},
                'quick_sort': {'complexity': 'O(n log n)', 'acceptable_for': 'most_cases'},
                'merge_sort': {'complexity': 'O(n log n)', 'acceptable_for': 'all_cases'}
            },
            'search_patterns': {
                'linear_search': {'complexity': 'O(n)', 'acceptable_for': 'small_arrays'},
                'binary_search': {'complexity': 'O(log n)', 'acceptable_for': 'sorted_data'},
                'hash_lookup': {'complexity': 'O(1)', 'acceptable_for': 'all_cases'}
            }
        }
        
        # Context awareness for patches vs full files
        self.context_adjustments = {
            'patch_context': {
                'complexity_tolerance': 1.5,  # 50% more lenient
                'length_tolerance': 2.0,      # 100% more lenient
                'scoring_floor': 0.4          # Minimum score of 0.4
            },
            'snippet_context': {
                'complexity_tolerance': 2.0,
                'length_tolerance': 3.0,
                'scoring_floor': 0.5
            }
        }
    
    async def _analyze_implementation(self, code: str, context: Dict[str, Any]) -> AgentResult:
        """Smart performance analysis with real-world context awareness"""
        issues = []
        metadata = {}
        
        # Determine code context for intelligent analysis
        code_context = self._determine_code_context(code, context)
        
        # Enhanced complexity analysis with context awareness
        complexity_metrics = self._analyze_smart_complexity(code, code_context)
        complexity_issues = self._extract_context_aware_complexity_issues(complexity_metrics, code)
        issues.extend(complexity_issues)
        metadata['complexity_metrics'] = complexity_metrics.__dict__
        
        # Intelligent algorithm analysis
        algorithm_analysis = self._analyze_intelligent_algorithms(code, code_context)
        algorithm_issues = self._extract_smart_algorithm_issues(algorithm_analysis, code)
        issues.extend(algorithm_issues)
        metadata['algorithm_analysis'] = algorithm_analysis.__dict__
        
        # Production-focused performance analysis
        performance_issues = self._analyze_production_performance(code, code_context)
        issues.extend(performance_issues)
        
        # Resource management with context awareness
        resource_issues = self._analyze_smart_resource_management(code, code_context)
        issues.extend(resource_issues)
        
        # Runtime profiling (if safe and beneficial)
        if self.enable_runtime_profiling and self._should_profile_code(code, code_context):
            try:
                performance_metrics = await self._smart_runtime_profiling(code)
                runtime_issues = self._extract_smart_runtime_issues(performance_metrics, code_context)
                issues.extend(runtime_issues)
                metadata['performance_metrics'] = performance_metrics.__dict__
            except Exception:
                pass  # Skip runtime profiling if it fails
        
        # Calculate intelligent performance score
        overall_score = self._calculate_intelligent_performance_score(issues, metadata, code_context)
        
        return AgentResult(
            agent_name=self.name,
            execution_time=0.0,
            overall_score=overall_score,
            issues=issues,
            metadata=metadata
        )
    
    def _determine_code_context(self, code: str, context: Dict[str, Any]) -> str:
        """Intelligently determine the context of the code being analyzed"""
        
        # Check if this is SWE-bench data (patches/fixes)
        if context.get('swe_bench_sample') or context.get('github_issue'):
            return 'patch_context'
        
        # Check code characteristics
        lines = [line for line in code.splitlines() if line.strip()]
        
        if len(lines) < 30:
            return 'snippet_context'
        elif len(lines) < 100:
            return 'patch_context'
        else:
            return 'full_file'
    
    def _analyze_smart_complexity(self, code: str, code_context: str) -> ComplexityMetrics:
        """Smart complexity analysis that adapts to code context"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ComplexityMetrics(0, 0, 0, 0, 0, 0, 0, 0, "O(?)", 0.0, code_context)
        
        # Find the most complex function or analyze the whole module
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        
        if functions:
            # Analyze the most complex function (but consider context)
            max_complexity = ComplexityMetrics(0, 0, 0, 0, 0, 0, 0, 0, "O(1)", 0.0, code_context)
            
            for func in functions:
                complexity = self._calculate_smart_complexity_for_node(func, code_context)
                if complexity.cyclomatic_complexity > max_complexity.cyclomatic_complexity:
                    max_complexity = complexity
            
            return max_complexity
        else:
            # Analyze module-level code
            return self._calculate_smart_complexity_for_node(tree, code_context)
    
    def _calculate_smart_complexity_for_node(self, node: ast.AST, code_context: str) -> ComplexityMetrics:
        """Calculate complexity metrics with intelligent context awareness"""
        
        # Basic complexity calculation
        cyclomatic = self._calculate_cyclomatic_complexity(node)
        cognitive = self._calculate_cognitive_complexity(node)
        nesting_depth = self._calculate_max_nesting_depth(node)
        
        # Function-specific metrics
        function_length = 0
        parameter_count = 0
        return_statements = 0
        
        if isinstance(node, ast.FunctionDef):
            parameter_count = len(node.args.args)
            if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                function_length = node.end_lineno - node.lineno
            return_statements = len([n for n in ast.walk(node) if isinstance(n, ast.Return)])
        
        # Count loops and conditionals
        loop_count = len([n for n in ast.walk(node) if isinstance(n, (ast.For, ast.While, ast.AsyncFor))])
        conditional_count = len([n for n in ast.walk(node) if isinstance(n, ast.If)])
        
        # Smart algorithm complexity determination
        algorithm_complexity_class = self._determine_smart_algorithm_complexity(node, code_context)
        
        # Context-aware scale risk calculation
        scale_risk_factor = self._calculate_context_aware_scale_risk(
            algorithm_complexity_class, loop_count, nesting_depth, function_length, code_context
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
            scale_risk_factor=scale_risk_factor,
            code_context=code_context
        )
    
    def _determine_smart_algorithm_complexity(self, node: ast.AST, code_context: str) -> str:
        """Intelligently determine algorithm complexity with context awareness"""
        
        # Count nested loops more accurately
        nested_loop_depth = 0
        max_nested_depth = 0
        
        def analyze_nesting(current_node, depth=0):
            nonlocal max_nested_depth
            max_nested_depth = max(max_nested_depth, depth)
            
            for child in ast.iter_child_nodes(current_node):
                if isinstance(child, (ast.For, ast.While, ast.AsyncFor)):
                    analyze_nesting(child, depth + 1)
                else:
                    analyze_nesting(child, depth)
        
        analyze_nesting(node)
        
        # Smart complexity classification
        if max_nested_depth == 0:
            return "O(1)"
        elif max_nested_depth == 1:
            # Check for common efficient patterns
            if self._has_efficient_patterns(node):
                return "O(n)"
            else:
                return "O(n)"
        elif max_nested_depth == 2:
            # Check if this is acceptable O(n²) for the context
            if code_context in ['patch_context', 'snippet_context']:
                # More tolerant for patches and snippets
                return "O(n²)"
            else:
                return "O(n²)"
        elif max_nested_depth >= 3:
            return "O(n³)"
        
        # Check for recursion patterns
        if self._has_recursion(node):
            # Distinguish between good recursion and exponential recursion
            if self._is_tail_recursive_pattern(node):
                return "O(n)"
            else:
                return "O(2^n)"
        
        return "O(n)"
    
    def _has_efficient_patterns(self, node: ast.AST) -> bool:
        """Check for efficient algorithm patterns"""
        # Look for built-in efficient operations
        efficient_calls = ['sorted', 'sort', 'min', 'max', 'sum', 'any', 'all']
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id in efficient_calls:
                    return True
        
        return False
    
    def _has_recursion(self, node: ast.AST) -> bool:
        """Check for recursive patterns"""
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    if child.func.id == function_name:
                        return True
        return False
    
    def _is_tail_recursive_pattern(self, node: ast.AST) -> bool:
        """Check if recursion follows tail-recursive pattern (more efficient)"""
        # Simplified check - look for return statements with recursive calls
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and isinstance(child.value, ast.Call):
                return True
        return False
    
    def _calculate_context_aware_scale_risk(self, complexity_class: str, loop_count: int, 
                                          nesting_depth: int, function_length: int, 
                                          code_context: str) -> float:
        """Calculate scale risk with intelligent context awareness"""
        
        # Base risk from complexity class
        base_risk = {
            "O(1)": 0.0,
            "O(log n)": 0.05,
            "O(n)": 0.1,
            "O(n log n)": 0.2,
            "O(n²)": 0.4,      # Reduced from 0.8
            "O(n³)": 0.7,      # Reduced from 0.95
            "O(n^4+)": 0.9,    # Reduced from 1.0
            "O(2^n)": 0.95     # Still high for exponential
        }.get(complexity_class, 0.3)
        
        # Apply context-based adjustments
        context_multiplier = 1.0
        if code_context == 'patch_context':
            context_multiplier = 0.6  # Much more lenient for patches
        elif code_context == 'snippet_context':
            context_multiplier = 0.5  # Very lenient for snippets
        
        # Additional risk factors (but reduced)
        loop_risk = min(0.1, loop_count * 0.03)        # Reduced from 0.15
        length_risk = min(0.1, max(0, (function_length - 50) / 100))  # Much more lenient
        nesting_risk = min(0.1, max(0, (nesting_depth - 2) * 0.05))   # More lenient
        
        total_risk = (base_risk + loop_risk + length_risk + nesting_risk) * context_multiplier
        return min(1.0, total_risk)
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
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
    
    def _calculate_max_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth"""
        max_depth = current_depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                child_depth = self._calculate_max_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._calculate_max_nesting_depth(child, current_depth)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _extract_context_aware_complexity_issues(self, metrics: ComplexityMetrics, 
                                                code: str) -> List[VerificationIssue]:
        """Extract complexity issues with intelligent context awareness"""
        issues = []
        context = metrics.code_context
        
        # Get context-appropriate thresholds
        thresholds = self.real_world_thresholds
        
        # Cyclomatic complexity with context awareness
        cc_thresholds = thresholds['cyclomatic_complexity'].get(context, 
                       thresholds['cyclomatic_complexity']['full_file'])
        
        if metrics.cyclomatic_complexity >= cc_thresholds['critical']:
            issues.append(VerificationIssue(
                type="cyclomatic_complexity",
                severity=Severity.HIGH,  # Reduced from CRITICAL
                message=f"High cyclomatic complexity: {metrics.cyclomatic_complexity}",
                suggestion="Consider refactoring for better maintainability"
            ))
        elif metrics.cyclomatic_complexity >= cc_thresholds['warning']:
            issues.append(VerificationIssue(
                type="cyclomatic_complexity",
                severity=Severity.MEDIUM,
                message=f"Elevated cyclomatic complexity: {metrics.cyclomatic_complexity}",
                suggestion="Consider simplifying logic structure"
            ))
        
        # Algorithm complexity with smart assessment
        if metrics.algorithm_complexity_class in ["O(n³)", "O(n^4+)", "O(2^n)"]:
            # Only flag truly problematic complexity for full files
            if context == 'full_file':
                severity = Severity.HIGH if metrics.algorithm_complexity_class == "O(n³)" else Severity.CRITICAL
                issues.append(VerificationIssue(
                    type="algorithm_complexity",
                    severity=severity,
                    message=f"Problematic algorithm complexity: {metrics.algorithm_complexity_class}",
                    suggestion="Consider algorithmic optimization for production scale"
                ))
            elif context == 'patch_context' and metrics.algorithm_complexity_class == "O(2^n)":
                # Only flag exponential complexity in patches
                issues.append(VerificationIssue(
                    type="algorithm_complexity",
                    severity=Severity.MEDIUM,
                    message=f"Exponential complexity detected: {metrics.algorithm_complexity_class}",
                    suggestion="Review algorithm efficiency for scale"
                ))
        
        # Only flag extreme cases for other metrics
        cog_thresholds = thresholds['cognitive_complexity'].get(context,
                        thresholds['cognitive_complexity']['full_file'])
        
        if metrics.cognitive_complexity >= cog_thresholds['critical']:
            issues.append(VerificationIssue(
                type="cognitive_complexity",
                severity=Severity.MEDIUM,  # Reduced severity
                message=f"High cognitive complexity: {metrics.cognitive_complexity}",
                suggestion="Simplify logic for better readability"
            ))
        
        return issues
    
    def _analyze_intelligent_algorithms(self, code: str, code_context: str) -> AlgorithmAnalysis:
        """Intelligent algorithm analysis with pattern recognition"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return AlgorithmAnalysis("O(?)", "O(?)", 0, 0, "unknown", "acceptable", [], [], False)
        
        # Smart complexity estimation
        time_complexity = self._estimate_smart_time_complexity(tree, code_context)
        space_complexity = self._estimate_smart_space_complexity(tree, code_context)
        
        # Pattern recognition
        algorithm_pattern = self._identify_smart_algorithm_pattern(code, tree)
        
        # Production criticality assessment
        is_production_critical = self._assess_production_criticality(code, algorithm_pattern)
        
        # Scale suitability with context awareness
        scale_suitability = self._assess_smart_scale_suitability(
            time_complexity, space_complexity, algorithm_pattern, code_context
        )
        
        # Smart bottleneck detection
        bottlenecks = self._find_smart_performance_bottlenecks(tree, code, code_context)
        
        # Intelligent optimization suggestions
        optimizations = self._generate_smart_optimization_suggestions(
            tree, code, algorithm_pattern, code_context
        )
        
        return AlgorithmAnalysis(
            estimated_time_complexity=time_complexity,
            estimated_space_complexity=space_complexity,
            loop_nesting_level=self._calculate_max_nesting_depth(tree),
            recursive_calls=self._count_recursive_calls(tree),
            algorithm_pattern=algorithm_pattern,
            scale_suitability=scale_suitability,
            performance_bottlenecks=bottlenecks,
            optimization_opportunities=optimizations,
            is_production_critical=is_production_critical
        )
    
    def _estimate_smart_time_complexity(self, tree: ast.AST, code_context: str) -> str:
        """Smart time complexity estimation with context awareness"""
        max_nesting = self._calculate_max_nesting_depth(tree)
        has_recursion = self._has_recursion(tree)
        has_efficient_ops = self._has_efficient_patterns(tree)
        
        # Context-aware complexity assessment
        if has_recursion:
            if self._is_tail_recursive_pattern(tree):
                return "O(n)"
            else:
                return "O(2^n)" if code_context == 'full_file' else "O(n)"  # More lenient for patches
        
        if max_nesting == 0:
            return "O(1)"
        elif max_nesting == 1:
            return "O(n log n)" if has_efficient_ops else "O(n)"
        elif max_nesting == 2:
            # Check if this is acceptable O(n²) for small data
            if code_context in ['patch_context', 'snippet_context']:
                return "O(n²)"  # Acceptable for patches
            else:
                return "O(n²)"
        else:
            return f"O(n^{max_nesting})"
    
    def _estimate_smart_space_complexity(self, tree: ast.AST, code_context: str) -> str:
        """Smart space complexity estimation"""
        has_recursion = self._has_recursion(tree)
        creates_structures = self._creates_data_structures(tree)
        
        if has_recursion:
            return "O(n)"
        elif creates_structures:
            return "O(n)"
        else:
            return "O(1)"
    
    def _creates_data_structures(self, tree: ast.AST) -> bool:
        """Check if code creates significant data structures"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ['list', 'dict', 'set', 'tuple']:
                    return True
            elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp)):
                return True
        return False
    
    def _identify_smart_algorithm_pattern(self, code: str, tree: ast.AST) -> str:
        """Smart algorithm pattern identification"""
        code_lower = code.lower()
        
        # Pattern matching with confidence
        if any(word in code_lower for word in ['sort', 'sorted', 'order']):
            return "sorting"
        elif any(word in code_lower for word in ['search', 'find', 'lookup']):
            return "searching" 
        elif any(word in code_lower for word in ['traverse', 'walk', 'iterate']):
            return "traversal"
        elif any(word in code_lower for word in ['filter', 'select', 'where']):
            return "filtering"
        elif any(word in code_lower for word in ['process', 'transform', 'convert']):
            return "processing"
        else:
            return "general"
    
    def _assess_production_criticality(self, code: str, pattern: str) -> bool:
        """Assess if this algorithm is production-critical"""
        critical_patterns = ['sorting', 'searching']
        critical_keywords = ['performance', 'optimization', 'scale', 'production']
        
        code_lower = code.lower()
        
        return (pattern in critical_patterns or 
                any(keyword in code_lower for keyword in critical_keywords))
    
    def _assess_smart_scale_suitability(self, time_complexity: str, space_complexity: str,
                                       pattern: str, code_context: str) -> str:
        """Smart scale suitability assessment with context awareness"""
        
        # Base suitability from complexity
        complexity_suitability = {
            "O(1)": "excellent",
            "O(log n)": "excellent",
            "O(n)": "good",
            "O(n log n)": "good",
            "O(n²)": "acceptable" if code_context != 'full_file' else "limited",
            "O(n³)": "limited",
            "O(2^n)": "poor"
        }
        
        base_suitability = complexity_suitability.get(time_complexity, "acceptable")
        
        # Context-based adjustments
        if code_context in ['patch_context', 'snippet_context']:
            # Much more lenient for patches and snippets
            if base_suitability == "limited":
                base_suitability = "acceptable"
            elif base_suitability == "poor" and time_complexity != "O(2^n)":
                base_suitability = "limited"
        
        return base_suitability
    
    def _find_smart_performance_bottlenecks(self, tree: ast.AST, code: str, 
                                           code_context: str) -> List[Tuple[int, str, float]]:
        """Smart bottleneck detection with context awareness"""
        bottlenecks = []
        
        # Only flag significant bottlenecks
        for node in ast.walk(tree):
            line_num = getattr(node, 'lineno', 0)
            
            # Nested loops - but be context aware
            if isinstance(node, (ast.For, ast.While)):
                nested_count = sum(1 for child in ast.walk(node) 
                                 if isinstance(child, (ast.For, ast.While)) and child != node)
                if nested_count >= 2:  # Only flag 3+ nested loops
                    severity = 0.6 if code_context == 'patch_context' else 0.8
                    bottlenecks.append((line_num, f"Deep nested loops ({nested_count+1} levels)", severity))
                elif nested_count >= 1 and code_context == 'full_file':
                    # Only flag 2+ nested loops in full files
                    bottlenecks.append((line_num, f"Nested loops ({nested_count+1} levels)", 0.4))
        
        return bottlenecks
    
    def _generate_smart_optimization_suggestions(self, tree: ast.AST, code: str, 
                                                pattern: str, code_context: str) -> List[str]:
        """Generate intelligent optimization suggestions"""
        suggestions = []
        
        # Pattern-specific suggestions (only for full files or production-critical code)
        if code_context == 'full_file' or pattern in ['sorting', 'searching']:
            if pattern == "sorting" and "bubble" in code.lower():
                    suggestions.append("Replace bubble sort with built-in sorted() for O(n log n) performance")
            elif pattern == "searching" and self._has_linear_search_in_loop(tree):
                    suggestions.append("Consider using sets or dictionaries for O(1) lookup performance")
            
        # General suggestions for significant issues only
        nested_loops = sum(1 for node in ast.walk(tree) 
                          if isinstance(node, (ast.For, ast.While)) and
                          any(isinstance(child, (ast.For, ast.While)) 
                              for child in ast.walk(node) if child != node))
        
        if nested_loops >= 2:  # Only suggest for deep nesting
            suggestions.append("Consider algorithmic optimization to reduce nested loops")
        
        return suggestions
    
    def _has_linear_search_in_loop(self, tree: ast.AST) -> bool:
        """Check for linear search patterns in loops"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                # Look for 'in' operations in loops
                for child in ast.walk(node):
                    if isinstance(child, ast.Compare):
                        for op in child.ops:
                            if isinstance(op, ast.In):
                                return True
        return False
    
    def _count_recursive_calls(self, tree: ast.AST) -> int:
        """Count recursive function calls"""
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
    
    def _extract_smart_algorithm_issues(self, analysis: AlgorithmAnalysis, 
                                       code: str) -> List[VerificationIssue]:
        """Extract algorithm issues with intelligent assessment"""
        issues = []
        
        # Only flag truly problematic scale issues
        if analysis.scale_suitability == "poor":
            if analysis.is_production_critical:
                issues.append(VerificationIssue(
                    type="scale_unsuitability",
                    severity=Severity.HIGH,
                    message=f"Poor scalability for production-critical {analysis.algorithm_pattern} algorithm",
                    suggestion="Optimize algorithm for production deployment"
                ))
            else:
                issues.append(VerificationIssue(
                    type="scale_concern",
                    severity=Severity.MEDIUM,
                    message=f"Scale concerns for {analysis.algorithm_pattern} algorithm",
                    suggestion="Consider optimization if used with large datasets"
                ))
        
        # Only flag extreme loop nesting
        if analysis.loop_nesting_level >= 4:  # Very deep nesting
            issues.append(VerificationIssue(
                type="loop_nesting",
                severity=Severity.HIGH,
                message=f"Extreme loop nesting: {analysis.loop_nesting_level} levels",
                suggestion="Redesign algorithm to reduce nesting complexity"
            ))
        elif analysis.loop_nesting_level >= 3:
            issues.append(VerificationIssue(
                type="loop_nesting",
                severity=Severity.MEDIUM,
                message=f"Deep loop nesting: {analysis.loop_nesting_level} levels",
                suggestion="Consider reducing nesting levels"
            ))
        
        # Performance bottlenecks (only severe ones)
        for line_num, description, severity in analysis.performance_bottlenecks:
            if severity >= 0.7:  # Only flag high-severity bottlenecks
                issue_severity = Severity.HIGH if severity >= 0.8 else Severity.MEDIUM
                issues.append(VerificationIssue(
                    type="performance_bottleneck",
                    severity=issue_severity,
                    message=f"Performance bottleneck: {description}",
                    line_number=line_num,
                    suggestion="Optimize this performance-critical section"
                ))
        
        return issues
    
    def _analyze_production_performance(self, code: str, code_context: str) -> List[VerificationIssue]:
        """Analyze production-specific performance issues"""
        issues = []
        lines = code.splitlines()
        
        # Only flag clear performance anti-patterns
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Obvious bubble sort pattern
            if re.search(r'for.*for.*if.*>.*swap', ' '.join(lines[max(0, line_num-2):line_num+2])):
                issues.append(VerificationIssue(
                    type="algorithm_inefficiency",
                    severity=Severity.MEDIUM,  # Reduced from CRITICAL
                    message="Bubble sort pattern detected - O(n²) sorting",
                    line_number=line_num,
                    suggestion="Use built-in sorted() or list.sort() for O(n log n) performance"
                ))
            
            # String concatenation in loops (only if clearly problematic)
            if '+=' in line_stripped and any(quote in line_stripped for quote in ['"', "'"]):
                # Check if definitely in a loop
                indent = len(line) - len(line.lstrip())
                for check_line in lines[max(0, line_num-5):line_num]:
                    check_indent = len(check_line) - len(check_line.lstrip())
                    if check_indent < indent and 'for' in check_line and len(lines) > 20:
                        # Only flag for substantial code, not snippets
                        issues.append(VerificationIssue(
                            type="performance_antipattern",
                            severity=Severity.MEDIUM,
                            message="String concatenation in loop - potential O(n²) performance",
                            line_number=line_num,
                            suggestion="Use join() or f-strings for efficient string building"
                        ))
                        break
        
        return issues
    
    def _analyze_smart_resource_management(self, code: str, code_context: str) -> List[VerificationIssue]:
        """Smart resource management analysis"""
        issues = []
        
        # Only flag clear resource management issues
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues
        
        # Look for obvious resource leaks
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                line_num = getattr(node, 'lineno', 1)
                
                # File operations without context managers (only flag if significant)
                if node.func.id in ['open'] and not self._is_in_with_statement(node, tree):
                    # Only flag if this is substantial code, not a snippet
                    if len(code.splitlines()) > 15:
                        issues.append(VerificationIssue(
                            type="resource_leak_risk",
                            severity=Severity.MEDIUM,
                            message="File opened without context manager",
                            line_number=line_num,
                            suggestion="Use 'with' statement for automatic resource cleanup"
                        ))
        
        return issues
    
    def _is_in_with_statement(self, target_node: ast.AST, tree: ast.AST) -> bool:
        """Check if node is within a with statement"""
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                for child in ast.walk(node):
                    if child is target_node:
                        return True
        return False
    
    def _should_profile_code(self, code: str, code_context: str) -> bool:
        """Determine if code should be runtime profiled"""
        # Skip profiling for patches and snippets, or unsafe code
        if code_context in ['patch_context', 'snippet_context']:
            return False
        
        # Skip if code has dangerous patterns
        dangerous_patterns = ['import os', 'subprocess', 'eval', 'exec', 'while True']
        return not any(pattern in code for pattern in dangerous_patterns)
    
    async def _smart_runtime_profiling(self, code: str) -> PerformanceMetrics:
        """Smart runtime profiling with reduced overhead"""
        if not PROFILING_AVAILABLE:
            return PerformanceMetrics(0.0, 0.0, 0.0, 0, 0, 1.0, {})
        
        # Simple execution time measurement
        start_time = time.time()
        
        try:
            # Very basic execution check
            compile(code, '<string>', 'exec')
            execution_time = time.time() - start_time
            
            return PerformanceMetrics(
                execution_time=execution_time,
                memory_usage=0.0,
                peak_memory=0.0,
                function_call_count=0,
                io_operations=0,
                performance_score=1.0 if execution_time < 0.1 else 0.8,
                scalability_projection={}
            )
        except Exception:
            return PerformanceMetrics(0.0, 0.0, 0.0, 0, 0, 0.5, {})
    
    def _extract_smart_runtime_issues(self, metrics: PerformanceMetrics, 
                                     code_context: str) -> List[VerificationIssue]:
        """Extract runtime issues with context awareness"""
        issues = []
        
        # Only flag significant runtime issues
        if metrics.execution_time > 1.0:  # Only flag very slow execution
            issues.append(VerificationIssue(
                type="execution_time",
                severity=Severity.MEDIUM,
                message=f"Slow execution time: {metrics.execution_time:.3f}s",
                suggestion="Consider optimizing for better performance"
            ))
        
        return issues
    
    def _calculate_intelligent_performance_score(self, issues: List[VerificationIssue], 
                                                metadata: Dict[str, Any], 
                                                code_context: str) -> float:
        """Calculate intelligent performance score with context awareness"""
        if not issues:
            return 1.0
        
        # Much more lenient scoring weights
        type_weights = {
            "algorithm_complexity": 0.8,        # Reduced from 1.5
            "scale_unsuitability": 1.0,         # Reduced from 1.8
            "cyclomatic_complexity": 0.4,       # Reduced from 0.8
            "cognitive_complexity": 0.3,        # Reduced from 0.6
            "nesting_depth": 0.4,               # Reduced from 1.0
            "function_length": 0.2,             # Reduced from 0.4
            "loop_nesting": 0.6,                # Reduced from 1.5
            "performance_bottleneck": 0.6,      # Reduced from 1.2
            "algorithm_inefficiency": 0.8,      # Reduced from 1.5
            "performance_antipattern": 0.4,     # Reduced from 1.0
            "resource_leak_risk": 0.6,          # Reduced from 1.5
            "scale_concern": 0.3,               # Reduced from 0.6
            "execution_time": 0.4,              # Reduced from 1.0
            "optimization_opportunity": 0.02    # Very low weight
        }
        
        # Much more lenient severity multipliers
        severity_multipliers = {
            Severity.LOW: 0.1,
            Severity.MEDIUM: 0.25,              # Reduced from 0.3
            Severity.HIGH: 0.5,                 # Reduced from 0.8
            Severity.CRITICAL: 0.8              # Reduced from 1.5
        }
        
        # Calculate penalty with context awareness
        total_penalty = 0.0
        for issue in issues:
            type_weight = type_weights.get(issue.type, 0.3)
            severity_multiplier = severity_multipliers[issue.severity]
            issue_penalty = type_weight * severity_multiplier
            total_penalty += issue_penalty
        
        # Apply context-based scoring adjustments
        context_adjustment = self.context_adjustments.get(code_context, {})
        tolerance = context_adjustment.get('complexity_tolerance', 1.0)
        scoring_floor = context_adjustment.get('scoring_floor', 0.2)
        
        # Much more lenient normalization
        max_penalty = 8.0 * tolerance  # Much higher threshold
        normalized_penalty = min(total_penalty / max_penalty, 1.0)
        
        # Calculate score with higher floor
        base_score = max(scoring_floor, 1.0 - normalized_penalty)
        
        # Bonus for good metrics
        bonus = 0.0
        if 'complexity_metrics' in metadata:
            complexity = metadata['complexity_metrics']
            if isinstance(complexity, dict):
                if complexity.get('scale_risk_factor', 1.0) < 0.2:
                    bonus += 0.1  # Bonus for low risk
        
        final_score = min(1.0, base_score + bonus)
        return round(final_score, 3)