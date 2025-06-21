# src/agents/style_maintainability.py
"""
Style & Maintainability Judge Agent - Code quality and maintainability assessment.
Analyzes code for style consistency, readability, documentation, and maintainability metrics.
"""

import ast
import re
import subprocess
import tempfile
import os
import sys
from typing import Dict, List, Any, Optional, Tuple, Set
import asyncio
from dataclasses import dataclass
from collections import defaultdict
import math

try:
    import black
    BLACK_AVAILABLE = True
except ImportError:
    BLACK_AVAILABLE = False

try:
    import flake8
    FLAKE8_AVAILABLE = True
except ImportError:
    FLAKE8_AVAILABLE = False

from .base_agent import BaseAgent, AgentResult, VerificationIssue, Severity


@dataclass
class StyleMetrics:
    """Style and formatting metrics"""
    line_length_violations: int
    indentation_inconsistencies: int
    naming_violations: int
    spacing_issues: int
    import_organization_score: float
    code_formatting_score: float


@dataclass
class DocumentationMetrics:
    """Documentation coverage metrics"""
    functions_with_docstrings: int
    classes_with_docstrings: int
    modules_with_docstrings: int
    total_functions: int
    total_classes: int
    docstring_coverage: float
    comment_density: float


@dataclass
class MaintainabilityMetrics:
    """Code maintainability metrics"""
    halstead_complexity: float
    maintainability_index: float
    code_duplication_score: float
    test_coverage_hints: int
    architectural_concerns: int
    refactoring_opportunities: int


@dataclass
class ReadabilityMetrics:
    """Code readability assessment"""
    variable_naming_score: float
    function_naming_score: float
    class_naming_score: float
    code_clarity_score: float
    logical_structure_score: float
    consistency_score: float


class StyleMaintainabilityJudge(BaseAgent):
    """
    Agent 4: Style & Maintainability Judge
    
    Performs comprehensive code quality analysis including:
    - Multi-language linting and style checking
    - Documentation coverage analysis  
    - Maintainability metrics calculation
    - Code readability assessment
    - Architectural pattern evaluation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("StyleMaintainabilityJudge", config)
        self.max_line_length = self.config.get('max_line_length', 88)
        self.min_docstring_coverage = self.config.get('min_docstring_coverage', 0.8)
        self.enable_external_linters = self.config.get('enable_external_linters', True)
        self.naming_conventions = self.config.get('naming_conventions', {
            'function': 'snake_case',
            'variable': 'snake_case', 
            'class': 'PascalCase',
            'constant': 'UPPER_CASE'
        })
        
        # Initialize style patterns
        self.style_patterns = self._initialize_style_patterns()
        self.naming_patterns = self._initialize_naming_patterns()
    
    def _initialize_style_patterns(self) -> Dict[str, str]:
        """Initialize style violation patterns"""
        return {
            'multiple_statements_per_line': r';\s*\w+',
            'trailing_whitespace': r'\s+$',
            'missing_whitespace_after_comma': r',\w',
            'missing_whitespace_around_operator': r'\w[+\-*/=]\w',
            'extra_blank_lines': r'\n\s*\n\s*\n',
            'inconsistent_quotes': r'("[^"]*".*\'[^\']*\'|\'[^\']*\'.*"[^"]*")',
            'space_before_comma': r'\w\s+,',
            'space_inside_brackets': r'[\(\[\{]\s+|\s+[\)\]\}]'
        }
    
    def _initialize_naming_patterns(self) -> Dict[str, str]:
        """Initialize naming convention patterns"""
        return {
            'snake_case': r'^[a-z][a-z0-9_]*$',
            'PascalCase': r'^[A-Z][a-zA-Z0-9]*$',
            'camelCase': r'^[a-z][a-zA-Z0-9]*$',
            'UPPER_CASE': r'^[A-Z][A-Z0-9_]*$',
            'SCREAMING_SNAKE_CASE': r'^[A-Z][A-Z0-9_]*$'
        }
    
    async def _analyze_implementation(self, code: str, context: Dict[str, Any]) -> AgentResult:
        """Main implementation of style and maintainability analysis"""
        issues = []
        metadata = {}
        
        # 1. Style Analysis
        style_metrics = self._analyze_style(code)
        style_issues = self._extract_style_issues(style_metrics, code)
        issues.extend(style_issues)
        metadata['style_metrics'] = style_metrics.__dict__
        
        # 2. Documentation Analysis
        doc_metrics = self._analyze_documentation(code)
        doc_issues = self._extract_documentation_issues(doc_metrics)
        issues.extend(doc_issues)
        metadata['documentation_metrics'] = doc_metrics.__dict__
        
        # 3. Maintainability Analysis
        maintainability_metrics = self._analyze_maintainability(code)
        maintainability_issues = self._extract_maintainability_issues(maintainability_metrics)
        issues.extend(maintainability_issues)
        metadata['maintainability_metrics'] = maintainability_metrics.__dict__
        
        # 4. Readability Analysis
        readability_metrics = self._analyze_readability(code)
        readability_issues = self._extract_readability_issues(readability_metrics)
        issues.extend(readability_issues)
        metadata['readability_metrics'] = readability_metrics.__dict__
        
        # 5. External Linter Integration
        if self.enable_external_linters:
            linter_issues = await self._run_external_linters(code)
            issues.extend(linter_issues)
        
        # 6. Architectural Pattern Analysis
        architectural_issues = self._analyze_architectural_patterns(code)
        issues.extend(architectural_issues)
        
        # Calculate overall quality score
        overall_score = self._calculate_quality_score(issues, metadata)
        
        return AgentResult(
            agent_name=self.name,
            execution_time=0.0,  # Will be set by base class
            overall_score=overall_score,
            issues=issues,
            metadata=metadata
        )
    
    def _analyze_style(self, code: str) -> StyleMetrics:
        """Analyze code style and formatting"""
        lines = code.splitlines()
        
        line_length_violations = 0
        indentation_inconsistencies = 0
        naming_violations = 0
        spacing_issues = 0
        
        # Analyze line lengths
        for line_num, line in enumerate(lines, 1):
            if len(line) > self.max_line_length:
                line_length_violations += 1
        
        # Analyze indentation consistency
        indentations = []
        for line in lines:
            if line.strip():  # Non-empty line
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    indentations.append(indent % 4)  # Check if multiple of 4
        
        if indentations:
            # Inconsistent if not all multiples of 4
            inconsistent_indents = sum(1 for indent in indentations if indent != 0)
            indentation_inconsistencies = inconsistent_indents
        
        # Analyze spacing issues
        for pattern_name, pattern in self.style_patterns.items():
            matches = re.findall(pattern, code, re.MULTILINE)
            spacing_issues += len(matches)
        
        # Calculate scores
        total_lines = len(lines) if lines else 1
        import_score = self._calculate_import_organization_score(code)
        formatting_score = max(0.0, 1.0 - (spacing_issues / max(total_lines, 1)) * 2)
        
        return StyleMetrics(
            line_length_violations=line_length_violations,
            indentation_inconsistencies=indentation_inconsistencies,
            naming_violations=naming_violations,
            spacing_issues=spacing_issues,
            import_organization_score=import_score,
            code_formatting_score=formatting_score
        )
    
    def _calculate_import_organization_score(self, code: str) -> float:
        """Calculate import organization score"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0.5
        
        imports = []
        import_positions = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(node)
                import_positions.append(getattr(node, 'lineno', 0))
        
        if not imports:
            return 1.0
        
        # Check if imports are at the top
        first_non_import = None
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom, ast.Module)):
                first_non_import = getattr(node, 'lineno', float('inf'))
                break
        
        imports_at_top = all(pos < first_non_import for pos in import_positions)
        
        # Check if imports are grouped (stdlib, third-party, local)
        # Simplified: just check if they're somewhat organized
        is_organized = import_positions == sorted(import_positions)
        
        score = 0.5  # Base score
        if imports_at_top:
            score += 0.3
        if is_organized:
            score += 0.2
        
        return min(1.0, score)
    
    def _extract_style_issues(self, metrics: StyleMetrics, code: str) -> List[VerificationIssue]:
        """Extract issues from style metrics"""
        issues = []
        
        # Line length violations
        if metrics.line_length_violations > 0:
            severity = Severity.MEDIUM if metrics.line_length_violations < 5 else Severity.HIGH
            issues.append(VerificationIssue(
                type="line_length",
                severity=severity,
                message=f"{metrics.line_length_violations} lines exceed {self.max_line_length} characters",
                suggestion=f"Keep lines under {self.max_line_length} characters for better readability"
            ))
        
        # Indentation issues
        if metrics.indentation_inconsistencies > 0:
            issues.append(VerificationIssue(
                type="indentation",
                severity=Severity.MEDIUM,
                message=f"{metrics.indentation_inconsistencies} indentation inconsistencies",
                suggestion="Use consistent 4-space indentation"
            ))
        
        # Spacing issues
        if metrics.spacing_issues > 0:
            severity = Severity.LOW if metrics.spacing_issues < 3 else Severity.MEDIUM
            issues.append(VerificationIssue(
                type="spacing",
                severity=severity,
                message=f"{metrics.spacing_issues} spacing/formatting issues",
                suggestion="Fix spacing around operators, commas, and brackets"
            ))
        
        # Import organization
        if metrics.import_organization_score < 0.7:
            issues.append(VerificationIssue(
                type="import_organization",
                severity=Severity.LOW,
                message=f"Poor import organization (score: {metrics.import_organization_score:.2f})",
                suggestion="Organize imports: stdlib, third-party, local modules"
            ))
        
        # Code formatting
        if metrics.code_formatting_score < 0.8:
            issues.append(VerificationIssue(
                type="code_formatting",
                severity=Severity.MEDIUM,
                message=f"Poor code formatting (score: {metrics.code_formatting_score:.2f})",
                suggestion="Consider using black or autopep8 for consistent formatting"
            ))
        
        return issues
    
    def _analyze_documentation(self, code: str) -> DocumentationMetrics:
        """Analyze documentation coverage"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return DocumentationMetrics(0, 0, 0, 0, 0, 0.0, 0.0)
        
        functions_with_docstrings = 0
        classes_with_docstrings = 0
        modules_with_docstrings = 0
        total_functions = 0
        total_classes = 0
        
        # Check module docstring
        if (isinstance(tree, ast.Module) and tree.body and 
            isinstance(tree.body[0], ast.Expr) and 
            isinstance(tree.body[0].value, ast.Constant) and
            isinstance(tree.body[0].value.value, str)):
            modules_with_docstrings = 1
        
        # Analyze functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                if self._has_docstring(node):
                    functions_with_docstrings += 1
            elif isinstance(node, ast.ClassDef):
                total_classes += 1
                if self._has_docstring(node):
                    classes_with_docstrings += 1
        
        # Calculate coverage
        total_documented = functions_with_docstrings + classes_with_docstrings + modules_with_docstrings
        total_documentable = total_functions + total_classes + 1  # +1 for module
        docstring_coverage = total_documented / total_documentable if total_documentable > 0 else 0.0
        
        # Calculate comment density
        lines = code.splitlines()
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        total_lines = len([line for line in lines if line.strip()])
        comment_density = comment_lines / total_lines if total_lines > 0 else 0.0
        
        return DocumentationMetrics(
            functions_with_docstrings=functions_with_docstrings,
            classes_with_docstrings=classes_with_docstrings,
            modules_with_docstrings=modules_with_docstrings,
            total_functions=total_functions,
            total_classes=total_classes,
            docstring_coverage=docstring_coverage,
            comment_density=comment_density
        )
    
    def _has_docstring(self, node: ast.AST) -> bool:
        """Check if a function or class has a docstring"""
        if (hasattr(node, 'body') and node.body and
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            return True
        return False
    
    def _extract_documentation_issues(self, metrics: DocumentationMetrics) -> List[VerificationIssue]:
        """Extract issues from documentation metrics"""
        issues = []
        
        # Docstring coverage
        if metrics.docstring_coverage < self.min_docstring_coverage:
            severity = Severity.HIGH if metrics.docstring_coverage < 0.5 else Severity.MEDIUM
            issues.append(VerificationIssue(
                type="docstring_coverage",
                severity=severity,
                message=f"Low docstring coverage: {metrics.docstring_coverage:.1%}",
                suggestion=f"Add docstrings to reach {self.min_docstring_coverage:.0%} coverage"
            ))
        
        # Missing function docstrings
        if metrics.total_functions > 0:
            missing_func_docs = metrics.total_functions - metrics.functions_with_docstrings
            if missing_func_docs > 0:
                issues.append(VerificationIssue(
                    type="missing_docstrings",
                    severity=Severity.MEDIUM,
                    message=f"{missing_func_docs} functions missing docstrings",
                    suggestion="Add docstrings to public functions"
                ))
        
        # Missing class docstrings
        if metrics.total_classes > 0:
            missing_class_docs = metrics.total_classes - metrics.classes_with_docstrings
            if missing_class_docs > 0:
                issues.append(VerificationIssue(
                    type="missing_docstrings",
                    severity=Severity.MEDIUM,
                    message=f"{missing_class_docs} classes missing docstrings",
                    suggestion="Add docstrings to classes"
                ))
        
        # Low comment density
        if metrics.comment_density < 0.1:
            issues.append(VerificationIssue(
                type="comment_density",
                severity=Severity.LOW,
                message=f"Low comment density: {metrics.comment_density:.1%}",
                suggestion="Add explanatory comments for complex logic"
            ))
        
        return issues
    
    def _analyze_maintainability(self, code: str) -> MaintainabilityMetrics:
        """Analyze code maintainability metrics"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return MaintainabilityMetrics(0.0, 0.0, 1.0, 0, 0, 0)
        
        # Calculate Halstead complexity (simplified)
        halstead = self._calculate_halstead_complexity(tree)
        
        # Calculate maintainability index (simplified)
        maintainability_index = self._calculate_maintainability_index(code, halstead)
        
        # Analyze code duplication
        duplication_score = self._analyze_code_duplication(code)
        
        # Count test coverage hints
        test_coverage_hints = self._count_test_coverage_hints(code)
        
        # Identify architectural concerns
        architectural_concerns = self._count_architectural_concerns(tree)
        
        # Identify refactoring opportunities
        refactoring_opportunities = self._count_refactoring_opportunities(tree)
        
        return MaintainabilityMetrics(
            halstead_complexity=halstead,
            maintainability_index=maintainability_index,
            code_duplication_score=duplication_score,
            test_coverage_hints=test_coverage_hints,
            architectural_concerns=architectural_concerns,
            refactoring_opportunities=refactoring_opportunities
        )
    
    def _calculate_halstead_complexity(self, tree: ast.AST) -> float:
        """Calculate simplified Halstead complexity"""
        operators = set()
        operands = set()
        
        for node in ast.walk(tree):
            # Count operators
            if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
                               ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
                               ast.And, ast.Or, ast.Not)):
                operators.add(type(node).__name__)
            
            # Count operands (simplified - just names and constants)
            elif isinstance(node, ast.Name):
                operands.add(node.id)
            elif isinstance(node, ast.Constant):
                operands.add(str(node.value))
        
        n1 = len(operators)  # Number of distinct operators
        n2 = len(operands)   # Number of distinct operands
        
        if n1 == 0 and n2 == 0:
            return 0.0
        
        # Simplified Halstead volume
        vocabulary = n1 + n2
        if vocabulary <= 1:
            return 0.0
        
        volume = (n1 + n2) * math.log2(vocabulary)
        return volume
    
    def _calculate_maintainability_index(self, code: str, halstead: float) -> float:
        """Calculate simplified maintainability index"""
        lines = [line for line in code.splitlines() if line.strip()]
        loc = len(lines)  # Lines of code
        
        if loc == 0:
            return 100.0
        
        # Simplified maintainability index
        # MI = 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(LOC)
        # Where HV = Halstead Volume, CC = Cyclomatic Complexity, LOC = Lines of Code
        
        # Estimate cyclomatic complexity
        cc = max(1, code.count('if') + code.count('for') + code.count('while') + 
                code.count('except') + code.count('elif'))
        
        if halstead <= 0:
            halstead = 1
        
        mi = 171 - 5.2 * math.log(halstead) - 0.23 * cc - 16.2 * math.log(loc)
        return max(0.0, min(100.0, mi))
    
    def _analyze_code_duplication(self, code: str) -> float:
        """Analyze code duplication (simplified)"""
        lines = [line.strip() for line in code.splitlines() if line.strip()]
        
        if len(lines) < 4:
            return 1.0  # No duplication possible
        
        # Look for repeated line sequences (simplified duplication detection)
        duplicated_lines = 0
        for i in range(len(lines) - 3):
            sequence = ' '.join(lines[i:i+3])
            for j in range(i + 3, len(lines) - 2):
                if ' '.join(lines[j:j+3]) == sequence:
                    duplicated_lines += 3
                    break
        
        duplication_ratio = duplicated_lines / len(lines) if lines else 0
        return max(0.0, 1.0 - duplication_ratio)
    
    def _count_test_coverage_hints(self, code: str) -> int:
        """Count hints about test coverage"""
        hints = 0
        
        # Look for test-related patterns
        test_patterns = [
            r'def test_',
            r'import unittest',
            r'import pytest',
            r'assert\s+',
            r'self\.assert',
            r'@pytest\.',
            r'TestCase'
        ]
        
        for pattern in test_patterns:
            hints += len(re.findall(pattern, code))
        
        return hints
    
    def _count_architectural_concerns(self, tree: ast.AST) -> int:
        """Count architectural concerns"""
        concerns = 0
        
        for node in ast.walk(tree):
            # God class detection (class with too many methods)
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                if len(methods) > 15:
                    concerns += 1
            
            # Long parameter lists
            elif isinstance(node, ast.FunctionDef):
                if len(node.args.args) > 8:
                    concerns += 1
                    
                # Method too long
                if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                    if node.end_lineno - node.lineno > 50:
                        concerns += 1
        
        return concerns
    
    def _count_refactoring_opportunities(self, tree: ast.AST) -> int:
        """Count refactoring opportunities"""
        opportunities = 0
        
        for node in ast.walk(tree):
            # Long if-elif chains
            if isinstance(node, ast.If):
                elif_count = 0
                current = node
                while hasattr(current, 'orelse') and current.orelse:
                    if (len(current.orelse) == 1 and 
                        isinstance(current.orelse[0], ast.If)):
                        elif_count += 1
                        current = current.orelse[0]
                    else:
                        break
                
                if elif_count > 5:
                    opportunities += 1
            
            # Nested loops (potential for optimization)
            elif isinstance(node, (ast.For, ast.While)):
                nested_loops = sum(1 for child in ast.walk(node) 
                                 if isinstance(child, (ast.For, ast.While)) and child != node)
                if nested_loops > 2:
                    opportunities += 1
        
        return opportunities
    
    def _extract_maintainability_issues(self, metrics: MaintainabilityMetrics) -> List[VerificationIssue]:
        """Extract issues from maintainability metrics"""
        issues = []
        
        # Low maintainability index
        if metrics.maintainability_index < 65:
            severity = Severity.HIGH if metrics.maintainability_index < 40 else Severity.MEDIUM
            issues.append(VerificationIssue(
                type="maintainability_index",
                severity=severity,
                message=f"Low maintainability index: {metrics.maintainability_index:.1f}",
                suggestion="Reduce complexity and improve code organization"
            ))
        
        # High Halstead complexity
        if metrics.halstead_complexity > 1000:
            issues.append(VerificationIssue(
                type="halstead_complexity",
                severity=Severity.MEDIUM,
                message=f"High Halstead complexity: {metrics.halstead_complexity:.1f}",
                suggestion="Simplify expressions and reduce operator usage"
            ))
        
        # Code duplication
        if metrics.code_duplication_score < 0.8:
            issues.append(VerificationIssue(
                type="code_duplication",
                severity=Severity.MEDIUM,
                message=f"Code duplication detected (score: {metrics.code_duplication_score:.2f})",
                suggestion="Extract duplicated code into reusable functions"
            ))
        
        # Architectural concerns
        if metrics.architectural_concerns > 0:
            issues.append(VerificationIssue(
                type="architectural_concerns",
                severity=Severity.MEDIUM,
                message=f"{metrics.architectural_concerns} architectural concerns found",
                suggestion="Review class and method design for better architecture"
            ))
        
        # Refactoring opportunities
        if metrics.refactoring_opportunities > 0:
            issues.append(VerificationIssue(
                type="refactoring_opportunities",
                severity=Severity.LOW,
                message=f"{metrics.refactoring_opportunities} refactoring opportunities",
                suggestion="Consider refactoring complex control structures"
            ))
        
        return issues
    
    def _analyze_readability(self, code: str) -> ReadabilityMetrics:
        """Analyze code readability"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ReadabilityMetrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        
        # Analyze naming conventions
        variable_naming_score = self._analyze_variable_naming(tree)
        function_naming_score = self._analyze_function_naming(tree)
        class_naming_score = self._analyze_class_naming(tree)
        
        # Analyze code clarity
        code_clarity_score = self._analyze_code_clarity(code, tree)
        
        # Analyze logical structure
        logical_structure_score = self._analyze_logical_structure(tree)
        
        # Analyze consistency
        consistency_score = self._analyze_consistency(code)
        
        return ReadabilityMetrics(
            variable_naming_score=variable_naming_score,
            function_naming_score=function_naming_score,
            class_naming_score=class_naming_score,
            code_clarity_score=code_clarity_score,
            logical_structure_score=logical_structure_score,
            consistency_score=consistency_score
        )
    
    def _analyze_variable_naming(self, tree: ast.AST) -> float:
        """Analyze variable naming conventions"""
        variables = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                variables.append(node.id)
        
        if not variables:
            return 1.0
        
        # Check naming convention compliance
        convention = self.naming_conventions.get('variable', 'snake_case')
        pattern = self.naming_patterns.get(convention, r'.*')
        
        compliant_vars = sum(1 for var in variables if re.match(pattern, var))
        return compliant_vars / len(variables)
    
    def _analyze_function_naming(self, tree: ast.AST) -> float:
        """Analyze function naming conventions"""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        if not functions:
            return 1.0
        
        # Check naming convention compliance
        convention = self.naming_conventions.get('function', 'snake_case')
        pattern = self.naming_patterns.get(convention, r'.*')
        
        compliant_funcs = sum(1 for func in functions if re.match(pattern, func))
        return compliant_funcs / len(functions)
    
    def _analyze_class_naming(self, tree: ast.AST) -> float:
        """Analyze class naming conventions"""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        if not classes:
            return 1.0
        
        # Check naming convention compliance
        convention = self.naming_conventions.get('class', 'PascalCase')
        pattern = self.naming_patterns.get(convention, r'.*')
        
        compliant_classes = sum(1 for cls in classes if re.match(pattern, cls))
        return compliant_classes / len(classes)
    
    def _analyze_logical_structure(self, tree: ast.AST) -> float:
        """Analyze logical structure and organization"""
        structure_score = 1.0
        
        # Check function ordering (simple heuristic)
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append((node.name, getattr(node, 'lineno', 0)))
        
        # Penalize if private functions come before public ones
        public_private_order_violations = 0
        for i, (name, _) in enumerate(functions):
            if not name.startswith('_'):  # Public function
                # Check if any private functions come after this
                for j, (later_name, _) in enumerate(functions[i+1:], i+1):
                    if later_name.startswith('_'):
                        public_private_order_violations += 1
                        break
        
        if len(functions) > 0:
            structure_score -= (public_private_order_violations / len(functions)) * 0.2
        
        return max(0.0, structure_score)
    
    def _analyze_consistency(self, code: str) -> float:
        """Analyze code consistency"""
        lines = code.splitlines()
        
        # Check quote consistency
        single_quotes = code.count("'")
        double_quotes = code.count('"')
        
        if single_quotes > 0 and double_quotes > 0:
            quote_consistency = 0.7  # Mixed quotes
        else:
            quote_consistency = 1.0
        
        # Check indentation consistency (already checked in style)
        indent_consistency = 1.0
        indent_sizes = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    indent_sizes.append(indent)
        
        if indent_sizes:
            # Check if all indents are multiples of the same base
            base_indent = min(indent_sizes) if indent_sizes else 4
            if base_indent > 0:
                inconsistent_indents = sum(1 for indent in indent_sizes if indent % base_indent != 0)
                indent_consistency = 1.0 - (inconsistent_indents / len(indent_sizes))
        
        # Average the consistency scores
        return (quote_consistency + indent_consistency) / 2
    
    def _extract_readability_issues(self, metrics: ReadabilityMetrics) -> List[VerificationIssue]:
        """Extract issues from readability metrics"""
        issues = []
        
        # Variable naming issues
        if metrics.variable_naming_score < 0.8:
            issues.append(VerificationIssue(
                type="variable_naming",
                severity=Severity.MEDIUM,
                message=f"Poor variable naming (score: {metrics.variable_naming_score:.2f})",
                suggestion=f"Use {self.naming_conventions.get('variable', 'snake_case')} for variables"
            ))
        
        # Function naming issues
        if metrics.function_naming_score < 0.8:
            issues.append(VerificationIssue(
                type="function_naming",
                severity=Severity.MEDIUM,
                message=f"Poor function naming (score: {metrics.function_naming_score:.2f})",
                suggestion=f"Use {self.naming_conventions.get('function', 'snake_case')} for functions"
            ))
        
        # Class naming issues
        if metrics.class_naming_score < 0.8:
            issues.append(VerificationIssue(
                type="class_naming",
                severity=Severity.MEDIUM,
                message=f"Poor class naming (score: {metrics.class_naming_score:.2f})",
                suggestion=f"Use {self.naming_conventions.get('class', 'PascalCase')} for classes"
            ))
        
        # Code clarity issues
        if metrics.code_clarity_score < 0.7:
            issues.append(VerificationIssue(
                type="code_clarity",
                severity=Severity.MEDIUM,
                message=f"Poor code clarity (score: {metrics.code_clarity_score:.2f})",
                suggestion="Use descriptive names and avoid magic numbers"
            ))
        
        # Logical structure issues
        if metrics.logical_structure_score < 0.8:
            issues.append(VerificationIssue(
                type="logical_structure",
                severity=Severity.LOW,
                message=f"Poor logical structure (score: {metrics.logical_structure_score:.2f})",
                suggestion="Organize functions and classes logically"
            ))
        
        # Consistency issues
        if metrics.consistency_score < 0.8:
            issues.append(VerificationIssue(
                type="consistency",
                severity=Severity.MEDIUM,
                message=f"Inconsistent code style (score: {metrics.consistency_score:.2f})",
                suggestion="Use consistent quotes, indentation, and formatting"
            ))
        
        return issues
    
    async def _run_external_linters(self, code: str) -> List[VerificationIssue]:
        """Run external linters if available"""
        issues = []
        
        # Black formatting check
        if BLACK_AVAILABLE:
            black_issues = await self._run_black_check(code)
            issues.extend(black_issues)
        
        # Flake8 style check
        if FLAKE8_AVAILABLE:
            flake8_issues = await self._run_flake8_check(code)
            issues.extend(flake8_issues)
        
        return issues
    
    async def _run_black_check(self, code: str) -> List[VerificationIssue]:
        """Check code formatting with Black"""
        try:
            import black
            
            # Try to format the code with Black
            formatted_code = black.format_str(code, mode=black.FileMode())
            
            if formatted_code != code:
                return [VerificationIssue(
                    type="black_formatting",
                    severity=Severity.LOW,
                    message="Code is not Black-formatted",
                    suggestion="Run 'black' to auto-format the code"
                )]
        except Exception:
            pass
        
        return []
    
    async def _run_flake8_check(self, code: str) -> List[VerificationIssue]:
        """Check code style with Flake8"""
        issues = []
        
        try:
            # Create temporary file for flake8
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Run flake8
                process = await asyncio.create_subprocess_exec(
                    sys.executable, '-m', 'flake8', temp_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    return issues  # No flake8 issues
                
                # Parse flake8 output
                flake8_output = stdout.decode()
                for line in flake8_output.splitlines():
                    if ':' in line:
                        parts = line.split(':', 3)
                        if len(parts) >= 4:
                            line_num = int(parts[1]) if parts[1].isdigit() else None
                            error_code = parts[3].strip().split()[0] if parts[3].strip() else ""
                            message = parts[3].strip()
                            
                            # Map flake8 codes to severities
                            severity = Severity.LOW
                            if error_code.startswith('E9') or error_code.startswith('F'):
                                severity = Severity.HIGH
                            elif error_code.startswith('E'):
                                severity = Severity.MEDIUM
                            
                            issues.append(VerificationIssue(
                                type="flake8",
                                severity=severity,
                                message=f"Flake8: {message}",
                                line_number=line_num,
                                suggestion="Fix style violations according to PEP 8"
                            ))
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
        
        except Exception:
            pass
        
        return issues
    
    def _analyze_architectural_patterns(self, code: str) -> List[VerificationIssue]:
        """Analyze architectural patterns and design principles"""
        issues = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues
        
        # Check for violation of Single Responsibility Principle
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Count different types of responsibilities
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                if len(methods) > 20:
                    issues.append(VerificationIssue(
                        type="srp_violation",
                        severity=Severity.MEDIUM,
                        message=f"Class '{node.name}' has too many methods ({len(methods)})",
                        suggestion="Consider splitting into multiple classes (Single Responsibility Principle)"
                    ))
        
        # Check for God functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                    function_length = node.end_lineno - node.lineno
                    if function_length > 100:
                        issues.append(VerificationIssue(
                            type="god_function",
                            severity=Severity.HIGH,
                            message=f"Function '{node.name}' is too long ({function_length} lines)",
                            suggestion="Break down into smaller, focused functions"
                        ))
        
        # Check for deep inheritance (simplified)
        class_inheritance = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                base_count = len(node.bases)
                if base_count > 3:
                    issues.append(VerificationIssue(
                        type="deep_inheritance",
                        severity=Severity.MEDIUM,
                        message=f"Class '{node.name}' has deep inheritance ({base_count} bases)",
                        suggestion="Consider composition over inheritance"
                    ))
        
        return issues
    
    def _calculate_quality_score(self, issues: List[VerificationIssue], 
                                metadata: Dict[str, Any]) -> float:
        """Calculate overall code quality score"""
        if not issues:
            return 1.0
        
        # Weight different types of quality issues
        type_weights = {
            "line_length": 0.3,
            "indentation": 0.4,
            "spacing": 0.3,
            "import_organization": 0.2,
            "code_formatting": 0.4,
            "docstring_coverage": 0.8,
            "missing_docstrings": 0.6,
            "comment_density": 0.2,
            "maintainability_index": 0.9,
            "halstead_complexity": 0.5,
            "code_duplication": 0.7,
            "architectural_concerns": 0.8,
            "refactoring_opportunities": 0.3,
            "variable_naming": 0.6,
            "function_naming": 0.6,
            "class_naming": 0.6,
            "code_clarity": 0.7,
            "logical_structure": 0.4,
            "consistency": 0.5,
            "black_formatting": 0.2,
            "flake8": 0.4,
            "srp_violation": 0.8,
            "god_function": 0.9,
            "deep_inheritance": 0.6
        }
        
        severity_multipliers = {
            Severity.LOW: 0.3,
            Severity.MEDIUM: 0.6,
            Severity.HIGH: 1.0,
            Severity.CRITICAL: 1.3
        }
        
        total_penalty = 0.0
        for issue in issues:
            type_weight = type_weights.get(issue.type, 0.5)
            severity_multiplier = severity_multipliers[issue.severity]
            issue_penalty = type_weight * severity_multiplier
            total_penalty += issue_penalty
        
        # Bonus for good documentation coverage
        doc_bonus = 0.0
        if 'documentation_metrics' in metadata:
            doc_metrics = metadata['documentation_metrics']
            if isinstance(doc_metrics, dict):
                coverage = doc_metrics.get('docstring_coverage', 0.0)
                if coverage > 0.8:
                    doc_bonus = 0.1 * (coverage - 0.8)
        
        # Bonus for high maintainability
        maintainability_bonus = 0.0
        if 'maintainability_metrics' in metadata:
            maint_metrics = metadata['maintainability_metrics']
            if isinstance(maint_metrics, dict):
                mi = maint_metrics.get('maintainability_index', 0.0)
                if mi > 80:
                    maintainability_bonus = 0.1 * ((mi - 80) / 20)
        
        # Normalize penalty (assuming 8 high-severity issues would give score 0)
        max_penalty = 8.0
        normalized_penalty = min(total_penalty / max_penalty, 1.0)
        
        base_score = max(0.0, 1.0 - normalized_penalty)
        return min(1.0, base_score + doc_bonus + maintainability_bonus)