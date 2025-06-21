# src/agents/correctness_critic.py
"""
Correctness Critic Agent - Enterprise-grade semantic correctness with edge case validation.
Implements production deployment standards with comprehensive exception path analysis.
"""

import ast
import subprocess
import tempfile
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import aiohttp
from dataclasses import dataclass

try:
    import tree_sitter
    import tree_sitter_python as tspython
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

try:
    from hypothesis import given, strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

from .base_agent import BaseAgent, AgentResult, VerificationIssue, Severity


@dataclass
class ASTMetrics:
    """Enhanced AST metrics with production focus"""
    cyclomatic_complexity: int
    nesting_depth: int
    function_count: int
    class_count: int
    potential_issues: List[str]
    import_count: int
    line_count: int
    exception_handling_coverage: float
    input_validation_score: float
    resource_safety_score: float


@dataclass
class SemanticAnalysis:
    """Enhanced semantic analysis with edge case detection"""
    logic_score: float
    clarity_score: float
    edge_case_coverage: float
    potential_bugs: List[str]
    suggestions: List[str]
    production_readiness: float


class CorrectnessAgent(BaseAgent):
    """
    Agent 1: Enterprise Correctness Critic
    
    Breakthrough features:
    - Exception path analysis for production deployment
    - Input validation detection with type safety assessment
    - Resource safety checks for memory/file leak prevention
    - Edge case detection with production impact analysis
    - Contract validation between function promises and implementation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("CorrectnessAgent", config)
        self.anthropic_api_key = self.config.get('anthropic_api_key')
        self.openai_api_key = self.config.get('openai_api_key')
        self.use_llm = self.config.get('use_llm', False)  # Disabled for MVP but enhanced analysis
        self.max_execution_time = self.config.get('max_execution_time', 5.0)
        
        # Enterprise correctness thresholds
        self.enterprise_thresholds = {
            'min_exception_coverage': 0.8,     # 80% exception handling required
            'min_input_validation': 0.7,       # 70% input validation required
            'max_function_complexity': 15,     # Max cyclomatic complexity
            'max_nesting_depth': 4,            # Max nesting levels
            'resource_safety_required': True   # Resource cleanup required
        }
        
        # Initialize tree-sitter if available
        self.ts_parser = None
        if TREE_SITTER_AVAILABLE:
            try:
                self.ts_parser = tree_sitter.Parser()
                self.ts_parser.set_language(tspython.language())
            except Exception:
                pass
    
    async def _analyze_implementation(self, code: str, context: Dict[str, Any]) -> AgentResult:
        """Enhanced correctness analysis with production deployment standards"""
        issues = []
        metadata = {}
        
        # 1. Enhanced AST Analysis with production focus
        ast_metrics = self._analyze_enhanced_ast(code)
        ast_issues = self._extract_enhanced_ast_issues(ast_metrics, code)
        issues.extend(ast_issues)
        metadata['ast_metrics'] = ast_metrics.__dict__
        
        # 2. Exception path analysis
        exception_issues = self._analyze_exception_paths(code)
        issues.extend(exception_issues)
        
        # 3. Input validation analysis
        input_validation_issues = self._analyze_input_validation(code)
        issues.extend(input_validation_issues)
        
        # 4. Resource safety analysis
        resource_safety_issues = self._analyze_resource_safety(code)
        issues.extend(resource_safety_issues)
        
        # 5. Edge case detection
        edge_case_issues = self._detect_edge_cases(code)
        issues.extend(edge_case_issues)
        
        # 6. Contract validation (function promises vs implementation)
        contract_issues = self._validate_function_contracts(code)
        issues.extend(contract_issues)
        
        # 7. Enhanced semantic analysis (simplified without LLM)
        semantic_analysis = self._enhanced_semantic_analysis(code)
        semantic_issues = self._extract_semantic_issues(semantic_analysis)
        issues.extend(semantic_issues)
        metadata['semantic_analysis'] = semantic_analysis.__dict__
        
        # 8. Safe execution validation
        execution_issues = await self._safe_execution_check(code)
        issues.extend(execution_issues)
        metadata['execution_validated'] = len(execution_issues) == 0
        
        # Calculate overall score with enterprise standards
        overall_score = self._calculate_enterprise_correctness_score(issues, metadata)
        
        return AgentResult(
            agent_name=self.name,
            execution_time=0.0,
            overall_score=overall_score,
            issues=issues,
            metadata=metadata
        )
    
    def _analyze_enhanced_ast(self, code: str) -> ASTMetrics:
        """Enhanced AST analysis with production deployment focus"""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ASTMetrics(
                cyclomatic_complexity=0,
                nesting_depth=0,
                function_count=0,
                class_count=0,
                potential_issues=[f"Syntax error: {str(e)}"],
                import_count=0,
                line_count=len(code.splitlines()),
                exception_handling_coverage=0.0,
                input_validation_score=0.0,
                resource_safety_score=0.0
            )
        
        # Enhanced AST analysis
        complexity = self._calculate_cyclomatic_complexity(tree)
        nesting_depth = self._calculate_nesting_depth(tree)
        function_count = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
        class_count = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
        import_count = len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])
        
        # Production-focused metrics
        exception_coverage = self._calculate_exception_coverage(tree)
        input_validation = self._calculate_input_validation_score(tree)
        resource_safety = self._calculate_resource_safety_score(tree)
        
        # Enhanced issue detection
        potential_issues = self._detect_enhanced_ast_issues(tree)
        
        return ASTMetrics(
            cyclomatic_complexity=complexity,
            nesting_depth=nesting_depth,
            function_count=function_count,
            class_count=class_count,
            potential_issues=potential_issues,
            import_count=import_count,
            line_count=len(code.splitlines()),
            exception_handling_coverage=exception_coverage,
            input_validation_score=input_validation,
            resource_safety_score=resource_safety
        )
    
    def _calculate_exception_coverage(self, tree: ast.AST) -> float:
        """Calculate percentage of functions with proper exception handling"""
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if not functions:
            return 1.0
        
        functions_with_exception_handling = 0
        for func in functions:
            has_try_except = any(isinstance(n, ast.Try) for n in ast.walk(func))
            has_input_validation = any(
                isinstance(n, ast.If) and self._is_validation_check(n) 
                for n in ast.walk(func)
            )
            
            if has_try_except or has_input_validation:
                functions_with_exception_handling += 1
        
        return functions_with_exception_handling / len(functions)
    
    def _is_validation_check(self, node: ast.If) -> bool:
        """Check if an if statement is likely input validation"""
        if isinstance(node.test, ast.Call):
            if isinstance(node.test.func, ast.Name):
                return node.test.func.id in ['isinstance', 'hasattr', 'callable']
        elif isinstance(node.test, ast.Compare):
            # Check for None comparisons, type checks, etc.
            return any(isinstance(op, (ast.Is, ast.IsNot, ast.In, ast.NotIn)) for op in node.test.ops)
        return False
    
    def _calculate_input_validation_score(self, tree: ast.AST) -> float:
        """Calculate input validation coverage score"""
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if not functions:
            return 1.0
        
        functions_with_validation = 0
        for func in functions:
            if len(func.args.args) == 0:  # No parameters to validate
                functions_with_validation += 1
                continue
            
            # Look for validation patterns in function body
            has_validation = False
            for node in ast.walk(func):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ['isinstance', 'hasattr', 'callable', 'len']:
                        has_validation = True
                        break
                elif isinstance(node, ast.If) and self._is_validation_check(node):
                    has_validation = True
                    break
                elif isinstance(node, ast.Raise):
                    # Explicit error raising suggests validation
                    has_validation = True
                    break
            
            if has_validation:
                functions_with_validation += 1
        
        return functions_with_validation / len(functions)
    
    def _calculate_resource_safety_score(self, tree: ast.AST) -> float:
        """Calculate resource safety score (proper cleanup, context managers)"""
        resource_operations = []
        safe_resource_operations = []
        
        for node in ast.walk(tree):
            # Identify resource operations
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ['open', 'socket', 'connect']:
                    resource_operations.append(node)
            
            # Identify safe resource usage (with statements)
            elif isinstance(node, ast.With):
                safe_resource_operations.append(node)
        
        if not resource_operations:
            return 1.0  # No resources to manage
        
        # Calculate ratio of safe to total resource operations
        return len(safe_resource_operations) / len(resource_operations)
    
    def _detect_enhanced_ast_issues(self, tree: ast.AST) -> List[str]:
        """Detect enhanced AST issues with production focus"""
        issues = []
        
        for node in ast.walk(tree):
            # Production deployment blockers
            if isinstance(node, ast.FunctionDef):
                # Function too long for maintainability
                if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                    func_length = node.end_lineno - node.lineno
                    if func_length > 50:
                        issues.append(f"Function '{node.name}' too long ({func_length} lines) - production maintainability concern")
                
                # Too many parameters
                if len(node.args.args) > 8:
                    issues.append(f"Function '{node.name}' has too many parameters ({len(node.args.args)}) - refactor needed")
                
                # Missing docstring for public functions
                if not node.name.startswith('_') and not self._has_docstring(node):
                    issues.append(f"Public function '{node.name}' missing docstring - production documentation required")
            
            # Dangerous exception handling patterns
            elif isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append("Bare except clause - production error handling must be specific")
            
            elif isinstance(node, ast.ExceptHandler):
                # Exception handling without logging or action
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    issues.append("Empty exception handler - production systems need error logging")
            
            # Resource management issues
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == 'open' and not self._is_in_with_statement(node, tree):
                    issues.append("File opened without 'with' statement - resource leak risk")
            
            # Global variable usage (production concern)
            elif isinstance(node, ast.Global):
                issues.append(f"Global variable usage: {', '.join(node.names)} - production state management concern")
            
            # Potential infinite loops
            elif isinstance(node, ast.While):
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    issues.append("Potential infinite loop with 'while True' - production stability risk")
        
        return issues
    
    def _has_docstring(self, node: ast.FunctionDef) -> bool:
        """Check if function has docstring"""
        if (node.body and
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            return True
        return False
    
    def _is_in_with_statement(self, target_node: ast.AST, tree: ast.AST) -> bool:
        """Check if a node is within a 'with' statement"""
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                for child in ast.walk(node):
                    if child is target_node:
                        return True
        return False
    
    def _analyze_exception_paths(self, code: str) -> List[VerificationIssue]:
        """Analyze exception handling paths for production readiness"""
        issues = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for functions that should have exception handling
                if self._should_have_exception_handling(node):
                    has_exception_handling = any(isinstance(n, ast.Try) for n in ast.walk(node))
                    if not has_exception_handling:
                        issues.append(VerificationIssue(
                            type="missing_exception_handling",
                            severity=Severity.HIGH,
                            message=f"Function '{node.name}' missing exception handling for production deployment",
                            line_number=getattr(node, 'lineno', None),
                            suggestion="Add try-except blocks for error conditions and resource cleanup"
                        ))
            
            elif isinstance(node, ast.ExceptHandler):
                # Check for proper exception handling
                if node.type is None:
                    issues.append(VerificationIssue(
                        type="bare_except",
                        severity=Severity.HIGH,
                        message="Bare except clause hides errors in production",
                        line_number=getattr(node, 'lineno', None),
                        suggestion="Specify exception types for proper error handling"
                    ))
                
                # Check for empty exception handlers
                elif len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    issues.append(VerificationIssue(
                        type="empty_exception_handler",
                        severity=Severity.MEDIUM,
                        message="Empty exception handler - production systems need error logging",
                        line_number=getattr(node, 'lineno', None),
                        suggestion="Add logging or appropriate error handling in exception blocks"
                    ))
        
        return issues
    
    def _should_have_exception_handling(self, func_node: ast.FunctionDef) -> bool:
        """Determine if function should have exception handling"""
        # Functions with file operations, network calls, or external dependencies
        risky_operations = ['open', 'request', 'connect', 'loads', 'dumps']
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in risky_operations:
                    return True
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in risky_operations:
                    return True
        
        return False
    
    def _analyze_input_validation(self, code: str) -> List[VerificationIssue]:
        """Analyze input validation for production safety"""
        issues = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip functions without parameters
                if len(node.args.args) == 0:
                    continue
                
                # Check for input validation
                has_validation = self._function_has_input_validation(node)
                if not has_validation and not node.name.startswith('_'):
                    issues.append(VerificationIssue(
                        type="missing_input_validation",
                        severity=Severity.MEDIUM,
                        message=f"Function '{node.name}' missing input validation for production safety",
                        line_number=getattr(node, 'lineno', None),
                        suggestion="Add type checking and value validation for function parameters"
                    ))
        
        return issues
    
    def _function_has_input_validation(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has input validation"""
        validation_patterns = [
            'isinstance', 'hasattr', 'callable', 'len', 'type',
            'ValueError', 'TypeError', 'AttributeError'
        ]
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in validation_patterns:
                    return True
            elif isinstance(node, ast.Raise):
                return True  # Explicit error raising suggests validation
            elif isinstance(node, ast.If) and self._is_validation_check(node):
                return True
        
        return False
    
    def _analyze_resource_safety(self, code: str) -> List[VerificationIssue]:
        """Analyze resource safety for production deployment"""
        issues = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues
        
        # Track resource operations
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ['open', 'socket'] and not self._is_in_with_statement(node, tree):
                    issues.append(VerificationIssue(
                        type="resource_leak_risk",
                        severity=Severity.HIGH,
                        message=f"Resource operation '{node.func.id}' without proper cleanup",
                        line_number=getattr(node, 'lineno', None),
                        suggestion="Use 'with' statement for automatic resource cleanup"
                    ))
        
        return issues
    
    def _detect_edge_cases(self, code: str) -> List[VerificationIssue]:
        """Detect missing edge case handling"""
        issues = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                edge_case_issues = self._check_function_edge_cases(node)
                issues.extend(edge_case_issues)
        
        return issues
    
    def _check_function_edge_cases(self, func_node: ast.FunctionDef) -> List[VerificationIssue]:
        """Check individual function for edge case handling"""
        issues = []
        
        # Check for common edge case patterns
        has_none_check = False
        has_empty_check = False
        has_boundary_check = False
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Compare):
                # Check for None comparisons
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant) and comparator.value is None:
                        has_none_check = True
                
                # Check for empty checks
                if any(isinstance(op, ast.Eq) for op in node.ops):
                    for comparator in node.comparators:
                        if isinstance(comparator, ast.Constant) and comparator.value in [0, '', []]:
                            has_empty_check = True
            
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == 'len':
                    has_boundary_check = True
        
        # Generate issues for missing edge cases
        if len(func_node.args.args) > 0 and not func_node.name.startswith('_'):
            if not has_none_check:
                issues.append(VerificationIssue(
                    type="missing_edge_case",
                    severity=Severity.MEDIUM,
                    message=f"Function '{func_node.name}' missing None value handling",
                    line_number=getattr(func_node, 'lineno', None),
                    suggestion="Add None checks for production robustness"
                ))
            
            # Check for functions that work with collections
            if self._function_works_with_collections(func_node) and not has_empty_check:
                issues.append(VerificationIssue(
                    type="missing_edge_case",
                    severity=Severity.MEDIUM,
                    message=f"Function '{func_node.name}' missing empty collection handling",
                    line_number=getattr(func_node, 'lineno', None),
                    suggestion="Add empty collection checks for production robustness"
                ))
        
        return issues
    
    def _function_works_with_collections(self, func_node: ast.FunctionDef) -> bool:
        """Check if function appears to work with collections"""
        collection_operations = ['append', 'extend', 'insert', 'remove', 'pop', 'index', 'count']
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Attribute) and node.attr in collection_operations:
                return True
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ['len', 'max', 'min', 'sum', 'sorted']:
                    return True
            elif isinstance(node, (ast.For, ast.ListComp, ast.DictComp)):
                return True
        
        return False
    
    def _validate_function_contracts(self, code: str) -> List[VerificationIssue]:
        """Validate function contracts (docstring promises vs implementation)"""
        issues = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                contract_issues = self._check_function_contract(node)
                issues.extend(contract_issues)
        
        return issues
    
    def _check_function_contract(self, func_node: ast.FunctionDef) -> List[VerificationIssue]:
        """Check if function implementation matches its contract"""
        issues = []
        
        # Get docstring if exists
        docstring = None
        if (func_node.body and
            isinstance(func_node.body[0], ast.Expr) and
            isinstance(func_node.body[0].value, ast.Constant) and
            isinstance(func_node.body[0].value.value, str)):
            docstring = func_node.body[0].value.value
        
        if docstring:
            # Check for common contract violations
            docstring_lower = docstring.lower()
            
            # Check if docstring mentions exceptions but none are raised
            if 'raises' in docstring_lower or 'exception' in docstring_lower:
                has_raises = any(isinstance(n, ast.Raise) for n in ast.walk(func_node))
                if not has_raises:
                    issues.append(VerificationIssue(
                        type="contract_violation",
                        severity=Severity.MEDIUM,
                        message=f"Function '{func_node.name}' docstring mentions exceptions but none are raised",
                        line_number=getattr(func_node, 'lineno', None),
                        suggestion="Ensure implementation matches documented behavior"
                    ))
            
            # Check if docstring mentions return type but function doesn't return
            if 'returns' in docstring_lower or 'return' in docstring_lower:
                has_return = any(isinstance(n, ast.Return) and n.value is not None 
                               for n in ast.walk(func_node))
                if not has_return:
                    issues.append(VerificationIssue(
                        type="contract_violation",
                        severity=Severity.MEDIUM,
                        message=f"Function '{func_node.name}' docstring mentions return value but function doesn't return",
                        line_number=getattr(func_node, 'lineno', None),
                        suggestion="Ensure function returns value as documented"
                    ))
        
        return issues
    
    def _enhanced_semantic_analysis(self, code: str) -> SemanticAnalysis:
        """Enhanced semantic analysis without LLM dependency"""
        potential_bugs = []
        suggestions = []
        
        # Basic semantic checks
        logic_score = 1.0
        clarity_score = 1.0
        edge_case_coverage = 0.8  # Default assumption
        
        try:
            tree = ast.parse(code)
            
            # Check for common logical issues
            for node in ast.walk(tree):
                if isinstance(node, ast.Compare):
                    # Check for potential logical errors
                    if len(node.ops) > 1:
                        # Chained comparisons can be confusing
                        clarity_score -= 0.1
                
                elif isinstance(node, ast.BoolOp):
                    # Complex boolean expressions
                    if len(node.values) > 3:
                        clarity_score -= 0.1
                        suggestions.append("Simplify complex boolean expressions")
                
                elif isinstance(node, ast.If):
                    # Check for missing else clauses in critical paths
                    if not node.orelse and self._is_critical_path(node):
                        potential_bugs.append("Missing else clause in critical decision path")
                        logic_score -= 0.1
            
            # Calculate edge case coverage
            functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            if functions:
                functions_with_edge_cases = sum(1 for f in functions if self._has_edge_case_handling(f))
                edge_case_coverage = functions_with_edge_cases / len(functions)
        
        except SyntaxError:
            logic_score = 0.5
            clarity_score = 0.5
            potential_bugs.append("Syntax errors prevent semantic analysis")
        
        # Calculate production readiness
        production_readiness = (logic_score + clarity_score + edge_case_coverage) / 3
        
        return SemanticAnalysis(
            logic_score=max(0.0, logic_score),
            clarity_score=max(0.0, clarity_score),
            edge_case_coverage=edge_case_coverage,
            potential_bugs=potential_bugs,
            suggestions=suggestions,
            production_readiness=production_readiness
        )
    
    def _is_critical_path(self, if_node: ast.If) -> bool:
        """Check if an if statement is in a critical execution path"""
        # Simple heuristic: if it contains return, raise, or assignment to important variables
        for node in ast.walk(if_node):
            if isinstance(node, (ast.Return, ast.Raise)):
                return True
            elif isinstance(node, ast.Assign):
                return True
        return False
    
    def _has_edge_case_handling(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has edge case handling"""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Compare):
                # Check for None, empty, or boundary checks
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant):
                        if comparator.value in [None, 0, '', []]:
                            return True
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ['len', 'isinstance', 'hasattr']:
                    return True
        return False
    
    def _extract_enhanced_ast_issues(self, metrics: ASTMetrics, code: str) -> List[VerificationIssue]:
        """Extract enhanced AST issues with enterprise focus"""
        issues = []
        
        # Check if this is simple, clean code (shouldn't have critical issues)
        lines = code.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        is_simple_code = len(non_empty_lines) < 30  # Simple code
        
        # Enterprise complexity thresholds (but not for simple clean code)
        if metrics.cyclomatic_complexity > self.enterprise_thresholds['max_function_complexity']:
            severity = Severity.MEDIUM if is_simple_code else Severity.HIGH  # Less harsh for simple code
            issues.append(VerificationIssue(
                type="complexity",
                severity=severity,
                message=f"Excessive cyclomatic complexity: {metrics.cyclomatic_complexity}",
                suggestion="Break function into smaller, focused functions for production maintainability"
            ))
        
        # Don't penalize simple code for missing documentation as heavily
        if metrics.exception_handling_coverage < self.enterprise_thresholds['min_exception_coverage']:
            # Simple code doesn't need enterprise-level exception handling
            if not is_simple_code:
                issues.append(VerificationIssue(
                    type="exception_coverage",
                    severity=Severity.HIGH,
                    message=f"Low exception handling coverage: {metrics.exception_handling_coverage:.1%}",
                    suggestion="Add exception handling for production deployment safety"
                ))
        
        # Only flag real production issues, not documentation in simple code
        for issue in metrics.potential_issues:
            severity = Severity.MEDIUM  # Default to medium, not critical
            
            if "Syntax error" in issue:
                severity = Severity.CRITICAL
            elif "infinite loop" in issue or "resource leak" in issue:
                severity = Severity.HIGH
            elif "production" in issue.lower() and not is_simple_code:
                severity = Severity.HIGH
            else:
                severity = Severity.LOW  # Most issues are low severity for clean code
            
            issues.append(VerificationIssue(
                type="ast_analysis",
                severity=severity,
                message=issue,
                suggestion="Address production deployment concern"
            ))
        
        return issues
        
    def _extract_semantic_issues(self, analysis: SemanticAnalysis) -> List[VerificationIssue]:
       """Extract issues from enhanced semantic analysis"""
       issues = []
       
       # Logic score issues (more aggressive thresholds)
       if analysis.logic_score < 0.7:
           issues.append(VerificationIssue(
               type="logic",
               severity=Severity.HIGH,
               message=f"Low logic score: {analysis.logic_score:.2f}",
               suggestion="Review code logic for production correctness"
           ))
       elif analysis.logic_score < 0.8:
           issues.append(VerificationIssue(
               type="logic",
               severity=Severity.MEDIUM,
               message=f"Moderate logic score: {analysis.logic_score:.2f}",
               suggestion="Consider improving code logic"
           ))
       
       # Edge case coverage issues
       if analysis.edge_case_coverage < 0.7:
           issues.append(VerificationIssue(
               type="edge_case_coverage",
               severity=Severity.HIGH,
               message=f"Low edge case coverage: {analysis.edge_case_coverage:.1%}",
               suggestion="Add edge case handling for production robustness"
           ))
       elif analysis.edge_case_coverage < 0.8:
           issues.append(VerificationIssue(
               type="edge_case_coverage",
               severity=Severity.MEDIUM,
               message=f"Moderate edge case coverage: {analysis.edge_case_coverage:.1%}",
               suggestion="Consider additional edge case handling"
           ))
       
       # Production readiness issues
       if analysis.production_readiness < 0.8:
           issues.append(VerificationIssue(
               type="production_readiness",
               severity=Severity.HIGH,
               message=f"Low production readiness: {analysis.production_readiness:.1%}",
               suggestion="Address correctness issues before production deployment"
           ))
       
       # Potential bugs
       for bug in analysis.potential_bugs:
           issues.append(VerificationIssue(
               type="potential_bug",
               severity=Severity.HIGH,
               message=f"Potential bug: {bug}",
               suggestion="Review and fix potential issue"
           ))
       
       return issues
   
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.With):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
   
    def _calculate_nesting_depth(self, tree: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth"""
        max_depth = current_depth
        
        for child in ast.iter_child_nodes(tree):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try, ast.FunctionDef, ast.ClassDef)):
                child_depth = self._calculate_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
   
    async def _safe_execution_check(self, code: str) -> List[VerificationIssue]:
        """Enhanced safe execution check"""
        issues = []
        
        # Enhanced safety checks
        dangerous_patterns = [
            'import os', 'import sys', 'import subprocess', 'import shutil',
            'open(', 'file(', 'exec(', 'eval(', '__import__',
            'input(', 'raw_input(', 'while True:', 'socket'
        ]
        
        if any(pattern in code for pattern in dangerous_patterns):
            issues.append(VerificationIssue(
                type="execution",
                severity=Severity.MEDIUM,
                message="Code contains potentially dangerous operations - skipping execution",
                suggestion="Manual review recommended for code with system operations"
            ))
            return issues
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                process = await asyncio.create_subprocess_exec(
                    sys.executable, '-c', f'exec(open("{temp_file}").read())',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), 
                        timeout=self.max_execution_time
                    )
                    
                    if process.returncode != 0:
                        error_msg = stderr.decode() if stderr else "Unknown execution error"
                        issues.append(VerificationIssue(
                            type="execution",
                            severity=Severity.HIGH,
                            message=f"Code execution failed: {error_msg.strip()}",
                            suggestion="Fix runtime errors before deployment"
                        ))
                
                except asyncio.TimeoutError:
                    process.kill()
                    issues.append(VerificationIssue(
                        type="execution",
                        severity=Severity.MEDIUM,
                        message="Code execution timed out",
                        suggestion="Check for infinite loops or long-running operations"
                    ))
            
            finally:
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
        
        except Exception as e:
            issues.append(VerificationIssue(
                type="execution",
                severity=Severity.MEDIUM,
                message=f"Could not execute code safely: {str(e)}",
                suggestion="Manual execution testing recommended"
            ))
        
        return issues
   
    def _calculate_enterprise_correctness_score(self, issues: List[VerificationIssue], 
                                            metadata: Dict[str, Any]) -> float:
        """Calculate enterprise correctness score with less aggressive penalties for good code"""
        if not issues:
            return 1.0
        
        # Less aggressive weights for correctness issues
        type_weights = {
            "complexity": 0.8,                  # Reduced from 1.2
            "nesting": 0.6,                     # Reduced from 1.0
            "exception_coverage": 1.0,          # Reduced from 1.5
            "input_validation": 0.8,            # Reduced from 1.2
            "resource_safety": 1.2,             # Reduced from 1.8
            "missing_exception_handling": 1.0,  # Reduced from 1.5
            "bare_except": 0.6,                 # Reduced from 1.0
            "empty_exception_handler": 0.4,     # Reduced from 0.8
            "missing_input_validation": 0.6,    # Reduced from 1.0
            "resource_leak_risk": 1.0,          # Reduced from 1.5
            "missing_edge_case": 0.6,           # Reduced from 1.0
            "contract_violation": 0.4,          # Reduced from 0.8
            "logic": 0.8,                       # Reduced from 1.3
            "edge_case_coverage": 0.6,          # Reduced from 1.0
            "production_readiness": 0.8,        # Reduced from 1.2
            "potential_bug": 0.8,               # Reduced from 1.3
            "ast_analysis": 0.4,                # Reduced from 0.8
            "execution": 0.6                    # Reduced from 1.0
        }
        
        # Less aggressive severity multipliers
        severity_multipliers = {
            Severity.LOW: 0.15,
            Severity.MEDIUM: 0.4,               # Reduced from 0.7
            Severity.HIGH: 0.8,                 # Reduced from 1.3
            Severity.CRITICAL: 1.5              # Reduced from 2.0
        }
        
        total_penalty = 0.0
        for issue in issues:
            type_weight = type_weights.get(issue.type, 0.6)
            severity_multiplier = severity_multipliers[issue.severity]
            issue_penalty = type_weight * severity_multiplier
            total_penalty += issue_penalty
        
        # Bonus for good metrics (kept same)
        bonus = 0.0
        if 'ast_metrics' in metadata:
            ast_metrics = metadata['ast_metrics']
            if isinstance(ast_metrics, dict):
                exception_coverage = ast_metrics.get('exception_handling_coverage', 0)
                if exception_coverage > 0.9:
                    bonus += 0.1
                
                input_validation = ast_metrics.get('input_validation_score', 0)
                if input_validation > 0.8:
                    bonus += 0.05
                
                resource_safety = ast_metrics.get('resource_safety_score', 0)
                if resource_safety > 0.9:
                    bonus += 0.1
        
        # Less aggressive normalization - higher max_penalty means gentler scoring
        max_penalty = 6.0  # Increased from 5.0 for gentler scoring
        normalized_penalty = min(total_penalty / max_penalty, 1.0)
        
        base_score = max(0.3, 1.0 - normalized_penalty)  # Higher floor from 0.0
        final_score = min(1.0, base_score + bonus)
        
        return final_score