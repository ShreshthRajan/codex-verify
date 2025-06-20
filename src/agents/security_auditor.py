# src/agents/security_auditor.py
"""
Security Auditor Agent - Comprehensive security vulnerability detection.
Analyzes code for security vulnerabilities, secrets, and unsafe patterns.
"""

import ast
import re
import hashlib
import math
import subprocess
import tempfile
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Set
import asyncio
import aiohttp
from dataclasses import dataclass
from urllib.parse import urlparse

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .base_agent import BaseAgent, AgentResult, VerificationIssue, Severity


@dataclass
class VulnerabilityPattern:
    """Security vulnerability pattern"""
    name: str
    pattern: str
    severity: Severity
    description: str
    cwe_id: Optional[str] = None


@dataclass
class SecurityMetrics:
    """Security analysis metrics"""
    vulnerability_count: int
    secrets_found: int
    unsafe_patterns: int
    dependency_issues: int
    total_risk_score: float
    categories: Dict[str, int]


@dataclass
class SecretCandidate:
    """Potential secret detected in code"""
    value: str
    type: str
    entropy: float
    line_number: int
    confidence: float


class SecurityAuditor(BaseAgent):
    """
    Agent 2: Security Auditor
    
    Performs comprehensive security analysis including:
    - Pattern-based vulnerability scanning
    - Secrets detection with entropy analysis
    - Dependency vulnerability checking
    - Code injection analysis
    - OWASP Top 10 coverage
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("SecurityAuditor", config)
        self.min_entropy_threshold = self.config.get('min_entropy_threshold', 4.5)
        self.check_dependencies = self.config.get('check_dependencies', True)
        self.safety_db_url = self.config.get('safety_db_url', 'https://pyup.io/api/v1/safety/')
        
        # Initialize vulnerability patterns
        self.vulnerability_patterns = self._initialize_vulnerability_patterns()
        self.secret_patterns = self._initialize_secret_patterns()
    
    def _initialize_vulnerability_patterns(self) -> List[VulnerabilityPattern]:
        """Initialize security vulnerability patterns"""
        return [
            # SQL Injection patterns
            VulnerabilityPattern(
                name="sql_injection",
                pattern=r'(execute|cursor\.execute|query)\s*\(\s*[\'"].*%.*[\'"]',
                severity=Severity.HIGH,
                description="Potential SQL injection vulnerability",
                cwe_id="CWE-89"
            ),
            VulnerabilityPattern(
                name="sql_format_string",
                pattern=r'(SELECT|INSERT|UPDATE|DELETE).*\.format\(',
                severity=Severity.HIGH,
                description="SQL query with string formatting",
                cwe_id="CWE-89"
            ),
            
            # Command Injection patterns
            VulnerabilityPattern(
                name="command_injection",
                pattern=r'(os\.system|subprocess\.call|subprocess\.run)\s*\(\s*.*\+',
                severity=Severity.HIGH,
                description="Potential command injection vulnerability",
                cwe_id="CWE-78"
            ),
            VulnerabilityPattern(
                name="shell_injection",
                pattern=r'shell\s*=\s*True',
                severity=Severity.MEDIUM,
                description="Shell execution enabled - potential injection risk",
                cwe_id="CWE-78"
            ),
            
            # Path Traversal patterns
            VulnerabilityPattern(
                name="path_traversal",
                pattern=r'open\s*\(\s*.*\+.*\)',
                severity=Severity.MEDIUM,
                description="Potential path traversal vulnerability",
                cwe_id="CWE-22"
            ),
            
            # Deserialization patterns
            VulnerabilityPattern(
                name="unsafe_deserialization",
                pattern=r'(pickle\.loads|pickle\.load|yaml\.load)\s*\(',
                severity=Severity.HIGH,
                description="Unsafe deserialization detected",
                cwe_id="CWE-502"
            ),
            
            # Eval/Exec patterns
            VulnerabilityPattern(
                name="code_injection",
                pattern=r'(eval|exec)\s*\(',
                severity=Severity.CRITICAL,
                description="Dynamic code execution detected",
                cwe_id="CWE-94"
            ),
            
            # Weak cryptography
            VulnerabilityPattern(
                name="weak_hash",
                pattern=r'hashlib\.(md5|sha1)\(',
                severity=Severity.MEDIUM,
                description="Weak cryptographic hash function",
                cwe_id="CWE-327"
            ),
            
            # LDAP Injection
            VulnerabilityPattern(
                name="ldap_injection",
                pattern=r'ldap.*search.*\+',
                severity=Severity.HIGH,
                description="Potential LDAP injection vulnerability",
                cwe_id="CWE-90"
            ),
            
            # XXE patterns
            VulnerabilityPattern(
                name="xxe_vulnerability",
                pattern=r'xml\.etree\.ElementTree\.parse.*input',
                severity=Severity.HIGH,
                description="Potential XXE vulnerability",
                cwe_id="CWE-611"
            ),
            
            # Hardcoded secrets patterns
            VulnerabilityPattern(
                name="hardcoded_password",
                pattern=r'(password|passwd|pwd)\s*=\s*[\'"][^\'\"]+[\'"]',
                severity=Severity.HIGH,
                description="Hardcoded password detected",
                cwe_id="CWE-798"
            ),
        ]
    
    def _initialize_secret_patterns(self) -> Dict[str, str]:
        """Initialize secret detection patterns"""
        return {
            'aws_access_key': r'AKIA[0-9A-Z]{16}',
            'aws_secret_key': r'[0-9a-zA-Z/+]{40}',
            'github_token': r'ghp_[0-9a-zA-Z]{36}',
            'slack_token': r'xox[baprs]-[0-9a-zA-Z\-]+',
            'stripe_key': r'sk_live_[0-9a-zA-Z]{24}',
            'openai_api_key': r'sk-[0-9a-zA-Z]{48}',
            'anthropic_api_key': r'sk-ant-[0-9a-zA-Z\-]+',
            'jwt_token': r'eyJ[0-9a-zA-Z_\-]+\.[0-9a-zA-Z_\-]+\.[0-9a-zA-Z_\-]+',
            'private_key': r'-----BEGIN (RSA |)PRIVATE KEY-----',
            'generic_api_key': r'[aA][pP][iI][_]?[kK][eE][yY].*[\'"][0-9a-zA-Z]{32,}[\'"]',
        }
    
    async def _analyze_implementation(self, code: str, context: Dict[str, Any]) -> AgentResult:
        """Main implementation of security analysis"""
        issues = []
        metadata = {}
        
        # 1. Vulnerability pattern scanning
        vuln_issues = self._scan_vulnerability_patterns(code)
        issues.extend(vuln_issues)
        
        # 2. Secrets detection
        secret_issues = self._detect_secrets(code)
        issues.extend(secret_issues)
        
        # 3. AST-based security analysis
        ast_security_issues = self._analyze_ast_security(code)
        issues.extend(ast_security_issues)
        
        # 4. Dependency vulnerability checking
        if self.check_dependencies and REQUESTS_AVAILABLE:
            dep_issues = await self._check_dependency_vulnerabilities(code)
            issues.extend(dep_issues)
        
        # 5. Taint analysis simulation
        taint_issues = self._simulate_taint_analysis(code)
        issues.extend(taint_issues)
        
        # Calculate security metrics
        security_metrics = self._calculate_security_metrics(issues)
        metadata['security_metrics'] = security_metrics.__dict__
        
        # Calculate overall score
        overall_score = self._calculate_security_score(issues, security_metrics)
        
        return AgentResult(
            agent_name=self.name,
            execution_time=0.0,  # Will be set by base class
            overall_score=overall_score,
            issues=issues,
            metadata=metadata
        )
    
    def _scan_vulnerability_patterns(self, code: str) -> List[VerificationIssue]:
        """Scan code for known vulnerability patterns"""
        issues = []
        lines = code.splitlines()
        
        for pattern_def in self.vulnerability_patterns:
            pattern = re.compile(pattern_def.pattern, re.IGNORECASE | re.MULTILINE)
            
            for line_num, line in enumerate(lines, 1):
                matches = pattern.findall(line)
                if matches:
                    issues.append(VerificationIssue(
                        type="vulnerability",
                        severity=pattern_def.severity,
                        message=f"{pattern_def.description}: {pattern_def.name}",
                        line_number=line_num,
                        suggestion=f"Review code for {pattern_def.name}. CWE: {pattern_def.cwe_id}",
                        confidence=0.8
                    ))
        
        return issues
    
    def _detect_secrets(self, code: str) -> List[VerificationIssue]:
        """Detect potential secrets using pattern matching and entropy analysis"""
        issues = []
        lines = code.splitlines()
        
        # Pattern-based secret detection
        for secret_type, pattern in self.secret_patterns.items():
            regex = re.compile(pattern)
            
            for line_num, line in enumerate(lines, 1):
                matches = regex.findall(line)
                for match in matches:
                    issues.append(VerificationIssue(
                        type="secret",
                        severity=Severity.HIGH,
                        message=f"Potential {secret_type} detected",
                        line_number=line_num,
                        suggestion=f"Remove hardcoded {secret_type} and use environment variables",
                        confidence=0.9
                    ))
        
        # Entropy-based detection
        entropy_secrets = self._detect_high_entropy_strings(code)
        for secret in entropy_secrets:
            issues.append(VerificationIssue(
                type="secret",
                severity=Severity.MEDIUM,
                message=f"High entropy string detected (entropy: {secret.entropy:.2f}): {secret.type}",
                line_number=secret.line_number,
                suggestion="Review if this is a hardcoded secret that should be externalized",
                confidence=secret.confidence
            ))
        
        return issues
    
    def _detect_high_entropy_strings(self, code: str) -> List[SecretCandidate]:
        """Detect high entropy strings that might be secrets"""
        candidates = []
        lines = code.splitlines()
        
        # Patterns for string assignments
        string_patterns = [
            r'[\'"]([a-zA-Z0-9+/=]{20,})[\'"]',  # Base64-like strings
            r'[\'"]([a-zA-Z0-9]{32,})[\'"]',     # Long alphanumeric strings
            r'[\'"]([a-f0-9]{32,})[\'"]',        # Hex strings
        ]
        
        for line_num, line in enumerate(lines, 1):
            for pattern in string_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    entropy = self._calculate_entropy(match)
                    
                    if entropy >= self.min_entropy_threshold:
                        # Determine string type
                        string_type = "unknown"
                        if re.match(r'^[a-zA-Z0-9+/]+=*$', match):
                            string_type = "base64_like"
                        elif re.match(r'^[a-f0-9]+$', match):
                            string_type = "hex_like"
                        elif re.match(r'^[a-zA-Z0-9]+$', match):
                            string_type = "alphanumeric"
                        
                        # Calculate confidence based on entropy and length
                        confidence = min(0.9, (entropy - 4.0) / 4.0 + 0.3)
                        
                        candidates.append(SecretCandidate(
                            value=match[:20] + "..." if len(match) > 20 else match,
                            type=string_type,
                            entropy=entropy,
                            line_number=line_num,
                            confidence=confidence
                        ))
        
        return candidates
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of a string"""
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        text_len = len(text)
        entropy = 0.0
        for count in char_counts.values():
            probability = count / text_len
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _analyze_ast_security(self, code: str) -> List[VerificationIssue]:
        """Perform AST-based security analysis"""
        issues = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues
        
        # Security-focused AST analysis
        for node in ast.walk(tree):
            # Check for dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['os', 'subprocess', 'sys', 'ctypes', 'importlib']:
                        issues.append(VerificationIssue(
                            type="dangerous_import",
                            severity=Severity.MEDIUM,
                            message=f"Potentially dangerous import: {alias.name}",
                            line_number=getattr(node, 'lineno', None),
                            suggestion=f"Review usage of {alias.name} module for security implications"
                        ))
            
            # Check for eval/exec calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile']:
                        issues.append(VerificationIssue(
                            type="code_execution",
                            severity=Severity.CRITICAL,
                            message=f"Dynamic code execution: {node.func.id}()",
                            line_number=getattr(node, 'lineno', None),
                            suggestion=f"Avoid using {node.func.id}() - major security risk"
                        ))
                    
                    elif node.func.id == 'open':
                        # Check for potential path traversal in open() calls
                        if len(node.args) > 0 and isinstance(node.args[0], ast.BinOp):
                            issues.append(VerificationIssue(
                                type="path_traversal",
                                severity=Severity.MEDIUM,
                                message="File path concatenation detected in open()",
                                line_number=getattr(node, 'lineno', None),
                                suggestion="Use os.path.join() or pathlib for safe path handling"
                            ))
            
            # Check for dangerous assignments
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Check for hardcoded credentials
                        if any(keyword in target.id.lower() for keyword in 
                               ['password', 'secret', 'key', 'token', 'api_key']):
                            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                                issues.append(VerificationIssue(
                                    type="hardcoded_secret",
                                    severity=Severity.HIGH,
                                    message=f"Potential hardcoded credential: {target.id}",
                                    line_number=getattr(node, 'lineno', None),
                                    suggestion="Use environment variables or secure configuration"
                                ))
        
        return issues
    
    async def _check_dependency_vulnerabilities(self, code: str) -> List[VerificationIssue]:
        """Check for known vulnerabilities in dependencies"""
        issues = []
        
        try:
            # Extract import statements
            tree = ast.parse(code)
            imports = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
            
            # Check against known vulnerable packages (simplified)
            known_vulnerable = {
                'pickle': "Pickle module can execute arbitrary code during deserialization",
                'marshal': "Marshal module can execute arbitrary code",
                'shelve': "Shelve uses pickle internally and inherits its vulnerabilities",
                'dill': "Dill can execute arbitrary code during deserialization",
            }
            
            for imp in imports:
                if imp in known_vulnerable:
                    issues.append(VerificationIssue(
                        type="vulnerable_dependency",
                        severity=Severity.MEDIUM,
                        message=f"Potentially vulnerable dependency: {imp}",
                        suggestion=f"Review usage: {known_vulnerable[imp]}"
                    ))
        
        except Exception:
            pass
        
        return issues
    
    def _simulate_taint_analysis(self, code: str) -> List[VerificationIssue]:
        """Simulate taint analysis for tracking untrusted data flow"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            # Track potential sources of untrusted data
            taint_sources = set()
            
            for node in ast.walk(tree):
                # Identify taint sources (user input, network, files)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ['input', 'raw_input']:
                        taint_sources.add('user_input')
                    elif node.func.id in ['open', 'read']:
                        taint_sources.add('file_input')
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['recv', 'read', 'get']:
                        taint_sources.add('network_input')
                
                # Check for potential sinks (SQL, system commands)
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['system', 'exec', 'eval']:
                            if taint_sources:
                                issues.append(VerificationIssue(
                                    type="taint_flow",
                                    severity=Severity.HIGH,
                                    message=f"Potential taint flow to {node.func.id}()",
                                    line_number=getattr(node, 'lineno', None),
                                    suggestion="Sanitize untrusted data before using in sensitive operations"
                                ))
        
        except Exception:
            pass
        
        return issues
    
    def _calculate_security_metrics(self, issues: List[VerificationIssue]) -> SecurityMetrics:
        """Calculate comprehensive security metrics"""
        vuln_count = len([i for i in issues if i.type == "vulnerability"])
        secrets_count = len([i for i in issues if i.type == "secret"])
        unsafe_patterns = len([i for i in issues if i.type in ["code_execution", "dangerous_import"]])
        dep_issues = len([i for i in issues if i.type == "vulnerable_dependency"])
        
        # Categorize issues
        categories = {}
        for issue in issues:
            category = issue.type
            categories[category] = categories.get(category, 0) + 1
        
        # Calculate total risk score
        severity_scores = {
            Severity.LOW: 1,
            Severity.MEDIUM: 3,
            Severity.HIGH: 7,
            Severity.CRITICAL: 10
        }
        
        total_risk_score = sum(severity_scores.get(issue.severity, 1) for issue in issues)
        
        return SecurityMetrics(
            vulnerability_count=vuln_count,
            secrets_found=secrets_count,
            unsafe_patterns=unsafe_patterns,
            dependency_issues=dep_issues,
            total_risk_score=total_risk_score,
            categories=categories
        )
    
    def _calculate_security_score(self, issues: List[VerificationIssue], 
                                 metrics: SecurityMetrics) -> float:
        """Calculate overall security score"""
        if not issues:
            return 1.0
        
        # Weight different types of security issues
        type_weights = {
            "vulnerability": 0.8,
            "secret": 0.9,
            "code_execution": 1.0,
            "dangerous_import": 0.4,
            "hardcoded_secret": 0.9,
            "taint_flow": 0.7,
            "vulnerable_dependency": 0.5
        }
        
        severity_multipliers = {
            Severity.LOW: 0.2,
            Severity.MEDIUM: 0.5,
            Severity.HIGH: 0.8,
            Severity.CRITICAL: 1.0
        }
        
        total_penalty = 0.0
        for issue in issues:
            type_weight = type_weights.get(issue.type, 0.5)
            severity_multiplier = severity_multipliers[issue.severity]
            issue_penalty = type_weight * severity_multiplier * issue.confidence
            total_penalty += issue_penalty
        
        # Normalize the score (assuming max 20 critical issues would give score 0)
        max_penalty = 20.0
        normalized_penalty = min(total_penalty / max_penalty, 1.0)
        
        return max(0.0, 1.0 - normalized_penalty)