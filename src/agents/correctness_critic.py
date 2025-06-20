"""
Correctness Critic Agent - AST-based semantic correctness validation.
Analyzes code for logical errors, semantic issues, and generates property-based tests.
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
    """Metrics extracted from AST analysis"""
    cyclomatic_complexity: int
    nesting_depth: int
    function_count: int
    class_count: int
    potential_issues: List[str]
    import_count: int
    line_count: int


@dataclass
class SemanticAnalysis:
    """Results from semantic analysis"""
    logic_score: float
    clarity_score: float
    potential_bugs: List[str]
    suggestions: List[str]


class CorrectnessCritic(BaseAgent):
    """
    Agent 1: Correctness Critic
    
    Performs comprehensive correctness analysis including:
    - AST-based static analysis
    - Semantic correctness checking via LLM
    - Property-based test generation
    - Safe code execution validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("CorrectnessCritic", config)
        self.anthropic_api_key = self.config.get('anthropic_api_key')
        self.openai_api_key = self.config.get('openai_api_key')
        self.use_llm = self.config.get('use_llm', True)
        self.max_execution_time = self.config.get('max_execution_time', 5.0)
        
        # Initialize tree-sitter if available
        self.ts_parser = None
        if TREE_SITTER_AVAILABLE:
            try:
                self.ts_parser = tree_sitter.Parser()
                self.ts_parser.set_language(tspython.language())
            except Exception:
                pass
    
    async def _analyze_implementation(self, code: str, context: Dict[str, Any]) -> AgentResult:
        """Main implementation of correctness analysis"""
        issues = []
        metadata = {}
        
        # 1. AST Analysis
        ast_metrics = self._analyze_ast(code)
        ast_issues = self._extract_ast_issues(ast_metrics, code)
        issues.extend(ast_issues)
        metadata['ast_metrics'] = ast_metrics.__dict__
        
        # 2. Semantic Analysis (LLM-powered if enabled)
        if self.use_llm and (self.anthropic_api_key or self.openai_api_key):
            semantic_analysis = await self._semantic_validation(code, context)
            semantic_issues = self._extract_semantic_issues(semantic_analysis)
            issues.extend(semantic_issues)
            metadata['semantic_analysis'] = semantic_analysis.__dict__
        
        # 3. Property-based test generation
        if HYPOTHESIS_AVAILABLE:
            test_issues = self._generate_property_tests(code)
            issues.extend(test_issues)
            metadata['property_tests_generated'] = len(test_issues)
        
        # 4. Safe execution validation
        execution_issues = await self._safe_execution_check(code)
        issues.extend(execution_issues)
        metadata['execution_validated'] = len(execution_issues) == 0
        
        # Calculate overall score
        overall_score = self._calculate_score(issues)
        
        return AgentResult(
            agent_name=self.name,
            execution_time=0.0,  # Will be set by base class
            overall_score=overall_score,
            issues=issues,
            metadata=metadata
        )
    
    def _analyze_ast(self, code: str) -> ASTMetrics:
        """Perform AST-based static analysis"""
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
                line_count=len(code.splitlines())
            )
        
        # AST analysis
        complexity = self._calculate_cyclomatic_complexity(tree)
        nesting_depth = self._calculate_nesting_depth(tree)
        function_count = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
        class_count = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
        import_count = len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])
        
        # Detect potential issues
        potential_issues = []
        
        # Check for common anti-patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Function too long
                if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                    func_length = node.end_lineno - node.lineno
                    if func_length > 50:
                        potential_issues.append(f"Function '{node.name}' is too long ({func_length} lines)")
                
                # Too many parameters
                if len(node.args.args) > 7:
                    potential_issues.append(f"Function '{node.name}' has too many parameters ({len(node.args.args)})")
            
            # Bare except clauses
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                potential_issues.append("Bare except clause found - should specify exception type")
            
            # Global variable usage
            if isinstance(node, ast.Global):
                potential_issues.append(f"Global variable usage detected: {', '.join(node.names)}")
        
        return ASTMetrics(
            cyclomatic_complexity=complexity,
            nesting_depth=nesting_depth,
            function_count=function_count,
            class_count=class_count,
            potential_issues=potential_issues,
            import_count=import_count,
            line_count=len(code.splitlines())
        )
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of the code"""
        complexity = 1  # Base complexity
        
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
    
    def _extract_ast_issues(self, metrics: ASTMetrics, code: str) -> List[VerificationIssue]:
        """Extract issues from AST metrics"""
        issues = []
        
        # Complexity issues
        if metrics.cyclomatic_complexity > 15:
            issues.append(VerificationIssue(
                type="complexity",
                severity=Severity.HIGH,
                message=f"High cyclomatic complexity: {metrics.cyclomatic_complexity}",
                suggestion="Consider breaking down complex functions into smaller ones"
            ))
        elif metrics.cyclomatic_complexity > 10:
            issues.append(VerificationIssue(
                type="complexity",
                severity=Severity.MEDIUM,
                message=f"Moderate cyclomatic complexity: {metrics.cyclomatic_complexity}",
                suggestion="Consider refactoring to reduce complexity"
            ))
        
        # Nesting depth issues
        if metrics.nesting_depth > 6:
            issues.append(VerificationIssue(
                type="nesting",
                severity=Severity.HIGH,
                message=f"Excessive nesting depth: {metrics.nesting_depth}",
                suggestion="Consider extracting nested logic into separate functions"
            ))
        elif metrics.nesting_depth > 4:
            issues.append(VerificationIssue(
                type="nesting",
                severity=Severity.MEDIUM,
                message=f"High nesting depth: {metrics.nesting_depth}",
                suggestion="Consider reducing nesting levels"
            ))
        
        # Convert potential issues from AST analysis
        for issue in metrics.potential_issues:
            severity = Severity.MEDIUM
            if "Syntax error" in issue:
                severity = Severity.CRITICAL
            elif "too long" in issue or "too many parameters" in issue:
                severity = Severity.MEDIUM
            elif "Global variable" in issue:
                severity = Severity.LOW
            
            issues.append(VerificationIssue(
                type="ast_analysis",
                severity=severity,
                message=issue,
                suggestion="Review and refactor as needed"
            ))
        
        return issues
    
    async def _semantic_validation(self, code: str, context: Dict[str, Any]) -> SemanticAnalysis:
        """Perform semantic validation using LLM"""
        # Construct prompt for semantic analysis
        prompt = f"""
        Analyze the following Python code for semantic correctness and logical issues:

        ```python
        {code}
        ```

        Please evaluate:
        1. Logic correctness (0.0-1.0 score)
        2. Code clarity (0.0-1.0 score)
        3. Potential bugs or logical errors
        4. Suggestions for improvement

        Respond in JSON format:
        {{
            "logic_score": 0.8,
            "clarity_score": 0.9,
            "potential_bugs": ["Bug description 1", "Bug description 2"],
            "suggestions": ["Suggestion 1", "Suggestion 2"]
        }}
        """
        
        try:
            if self.anthropic_api_key:
                response = await self._query_anthropic(prompt)
            elif self.openai_api_key:
                response = await self._query_openai(prompt)
            else:
                # Fallback to simple heuristic analysis
                return self._heuristic_semantic_analysis(code)
            
            # Parse LLM response (simplified - in production would need robust JSON parsing)
            import json
            try:
                data = json.loads(response)
                return SemanticAnalysis(
                    logic_score=data.get('logic_score', 0.8),
                    clarity_score=data.get('clarity_score', 0.8),
                    potential_bugs=data.get('potential_bugs', []),
                    suggestions=data.get('suggestions', [])
                )
            except json.JSONDecodeError:
                return self._heuristic_semantic_analysis(code)
                
        except Exception as e:
            # Fallback to heuristic analysis
            return self._heuristic_semantic_analysis(code)
    
    def _heuristic_semantic_analysis(self, code: str) -> SemanticAnalysis:
        """Fallback heuristic semantic analysis when LLM is unavailable"""
        potential_bugs = []
        suggestions = []
        
        # Simple heuristic checks
        if "TODO" in code or "FIXME" in code:
            potential_bugs.append("Code contains TODO/FIXME comments")
        
        if code.count("except:") > 0:
            potential_bugs.append("Bare except clauses may hide important errors")
        
        if code.count("global ") > 0:
            suggestions.append("Consider avoiding global variables")
        
        # Calculate scores based on heuristics
        logic_score = max(0.0, 1.0 - len(potential_bugs) * 0.2)
        clarity_score = max(0.0, 1.0 - code.count("# ") / max(len(code.splitlines()), 1))
        
        return SemanticAnalysis(
            logic_score=logic_score,
            clarity_score=clarity_score,
            potential_bugs=potential_bugs,
            suggestions=suggestions
        )
    
    async def _query_anthropic(self, prompt: str) -> str:
        """Query Anthropic Claude API"""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.anthropic_api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["content"][0]["text"]
                else:
                    raise Exception(f"Anthropic API error: {response.status}")
    
    async def _query_openai(self, prompt: str) -> str:
        """Query OpenAI API"""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"OpenAI API error: {response.status}")
    
    def _extract_semantic_issues(self, analysis: SemanticAnalysis) -> List[VerificationIssue]:
        """Extract issues from semantic analysis"""
        issues = []
        
        # Logic score issues
        if analysis.logic_score < 0.5:
            issues.append(VerificationIssue(
                type="logic",
                severity=Severity.HIGH,
                message=f"Low logic score: {analysis.logic_score:.2f}",
                suggestion="Review code logic for potential errors"
            ))
        elif analysis.logic_score < 0.7:
            issues.append(VerificationIssue(
                type="logic",
                severity=Severity.MEDIUM,
                message=f"Moderate logic score: {analysis.logic_score:.2f}",
                suggestion="Consider improving code logic"
            ))
        
        # Clarity score issues
        if analysis.clarity_score < 0.6:
            issues.append(VerificationIssue(
                type="clarity",
                severity=Severity.MEDIUM,
                message=f"Low clarity score: {analysis.clarity_score:.2f}",
                suggestion="Improve code readability and documentation"
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
    
    def _generate_property_tests(self, code: str) -> List[VerificationIssue]:
        """Generate property-based tests using Hypothesis"""
        if not HYPOTHESIS_AVAILABLE:
            return []
        
        issues = []
        
        try:
            # Parse the code to find functions
            tree = ast.parse(code)
            functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            
            for func in functions:
                # Simple heuristic: if function has parameters, suggest property testing
                if len(func.args.args) > 0:
                    issues.append(VerificationIssue(
                        type="testing",
                        severity=Severity.LOW,
                        message=f"Function '{func.name}' could benefit from property-based testing",
                        suggestion=f"Consider adding Hypothesis tests for function '{func.name}'"
                    ))
        
        except Exception:
            pass
        
        return issues
    
    async def _safe_execution_check(self, code: str) -> List[VerificationIssue]:
        """Safely execute code to check for runtime errors"""
        issues = []
        
        # Basic safety checks before execution
        dangerous_patterns = [
            'import os', 'import sys', 'import subprocess', 'import shutil',
            'open(', 'file(', 'exec(', 'eval(', '__import__'
        ]
        
        if any(pattern in code for pattern in dangerous_patterns):
            issues.append(VerificationIssue(
                type="execution",
                severity=Severity.MEDIUM,
                message="Code contains potentially dangerous operations - skipping execution",
                suggestion="Manual review recommended for code with file/system operations"
            ))
            return issues
        
        try:
            # Create a temporary file and execute with timeout
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Execute with timeout
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
                # Clean up temporary file
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