# src/agents/security_auditor.py
"""
Security Auditor Agent - Enterprise-grade compound vulnerability detection.
Implements context-aware security analysis with production deployment standards.
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
    """Enhanced security vulnerability pattern with context awareness"""
    name: str
    pattern: str
    severity: Severity
    description: str
    cwe_id: Optional[str] = None
    context_multipliers: Dict[str, float] = None  # Context-based severity multipliers


@dataclass
class SecurityMetrics:
    """Enhanced security analysis metrics"""
    vulnerability_count: int
    secrets_found: int
    unsafe_patterns: int
    dependency_issues: int
    total_risk_score: float
    compound_vulnerabilities: int
    categories: Dict[str, int]
    context_risks: Dict[str, float]


@dataclass
class SecretCandidate:
    """Enhanced secret detection with production impact"""
    value: str
    type: str
    entropy: float
    line_number: int
    confidence: float
    production_impact: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"


class SecurityAuditor(BaseAgent):
    """
    Agent 2: Enterprise Security Auditor
    
    Breakthrough features:
    - Compound vulnerability detection with exponential risk scoring
    - Context-aware severity escalation for production environments
    - Industry-standard security frameworks (OWASP Top 10, CWE mapping)
    - Crypto-intelligence with algorithm-specific risk assessment
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("SecurityAuditor", config)
        self.min_entropy_threshold = self.config.get('min_entropy_threshold', 3.5)  # More aggressive
        self.check_dependencies = self.config.get('check_dependencies', True)
        
        # Enterprise security thresholds
        self.enterprise_thresholds = {
            'max_secrets_per_file': 0,          # Zero tolerance for production
            'max_critical_vulnerabilities': 0,  # Zero tolerance for critical
            'max_high_vulnerabilities': 1,      # Very strict for high
            'crypto_compliance_required': True   # Must use approved algorithms
        }
        
        # Initialize enhanced patterns
        self.vulnerability_patterns = self._initialize_enhanced_patterns()
        self.secret_patterns = self._initialize_enhanced_secret_patterns()
        self.context_keywords = self._initialize_context_keywords()
    
    def _initialize_enhanced_patterns(self) -> List[VulnerabilityPattern]:
        """Initialize enhanced vulnerability patterns with context awareness"""
        return [
            # SQL Injection - Context-aware severity
            VulnerabilityPattern(
                name="sql_injection_direct",
                pattern=r'(execute|cursor\.execute|query)\s*\(\s*[\'"].*%.*[\'"]',
                severity=Severity.HIGH,
                description="Direct SQL injection vulnerability",
                cwe_id="CWE-89",
                context_multipliers={'auth': 2.0, 'user': 1.5, 'admin': 2.0}
            ),
            VulnerabilityPattern(
                name="sql_format_injection",
                pattern=r'(SELECT|INSERT|UPDATE|DELETE).*\.format\(',
                severity=Severity.HIGH,
                description="SQL injection via string formatting",
                cwe_id="CWE-89",
                context_multipliers={'password': 2.5, 'login': 2.0, 'payment': 3.0}
            ),
            VulnerabilityPattern(
                name="sql_fstring_injection", 
                pattern=r'(SELECT|INSERT|UPDATE|DELETE).*f[\'"].*\{.*\}.*[\'"]',
                severity=Severity.HIGH,
                description="SQL injection via f-string formatting",
                cwe_id="CWE-89",
                context_multipliers={'user_id': 2.0, 'account': 2.5}
            ),
            
            # Command Injection - Enhanced detection
            VulnerabilityPattern(
                name="command_injection_concat",
                pattern=r'(os\.system|subprocess\.call|subprocess\.run)\s*\(\s*.*\+',
                severity=Severity.HIGH,
                description="Command injection via string concatenation",
                cwe_id="CWE-78",
                context_multipliers={'user_input': 2.0, 'filename': 1.5}
            ),
            VulnerabilityPattern(
                name="shell_injection_enabled",
                pattern=r'shell\s*=\s*True',
                severity=Severity.MEDIUM,
                description="Shell execution enabled - injection vector",
                cwe_id="CWE-78",
                context_multipliers={'user': 1.5, 'input': 1.5}
            ),
            VulnerabilityPattern(
                name="os_system_direct",
                pattern=r'os\.system\s*\(',
                severity=Severity.HIGH,
                description="Direct os.system() usage - command injection risk",
                cwe_id="CWE-78"
            ),
            
            # Code Execution - Critical threats
            VulnerabilityPattern(
                name="eval_execution",
                pattern=r'eval\s*\(',
                severity=Severity.CRITICAL,
                description="eval() enables arbitrary code execution",
                cwe_id="CWE-94",
                context_multipliers={'user': 3.0, 'input': 3.0, 'request': 2.5}
            ),
            VulnerabilityPattern(
                name="exec_execution",
                pattern=r'exec\s*\(',
                severity=Severity.CRITICAL,
                description="exec() enables arbitrary code execution",
                cwe_id="CWE-94"
            ),
            VulnerabilityPattern(
                name="compile_execution",
                pattern=r'compile\s*\(\s*.*input',
                severity=Severity.CRITICAL,
                description="compile() with user input - code injection",
                cwe_id="CWE-94"
            ),
            
            # Deserialization - Enhanced detection
            VulnerabilityPattern(
                name="pickle_loads_unsafe",
                pattern=r'pickle\.loads\s*\(',
                severity=Severity.HIGH,
                description="pickle.loads() can execute arbitrary code",
                cwe_id="CWE-502",
                context_multipliers={'request': 2.0, 'network': 2.0, 'api': 1.5}
            ),
            VulnerabilityPattern(
                name="yaml_load_unsafe",
                pattern=r'yaml\.load\s*\(',
                severity=Severity.HIGH,
                description="yaml.load() enables code execution",
                cwe_id="CWE-502"
            ),
            VulnerabilityPattern(
                name="marshal_loads_unsafe",
                pattern=r'marshal\.loads\s*\(',
                severity=Severity.HIGH,
                description="marshal.loads() can execute arbitrary code",
                cwe_id="CWE-502"
            ),
            
            # Cryptographic Vulnerabilities - Enhanced
            VulnerabilityPattern(
                name="md5_usage",
                pattern=r'hashlib\.md5\s*\(',
                severity=Severity.MEDIUM,
                description="MD5 is cryptographically broken",
                cwe_id="CWE-327",
                context_multipliers={'password': 3.0, 'token': 2.5, 'auth': 2.0}
            ),
            VulnerabilityPattern(
                name="sha1_usage",
                pattern=r'hashlib\.sha1\s*\(',
                severity=Severity.MEDIUM,
                description="SHA1 is cryptographically weak",
                cwe_id="CWE-327",
                context_multipliers={'password': 2.5, 'signature': 2.0}
            ),
            VulnerabilityPattern(
                name="weak_random",
                pattern=r'random\.(randint|choice|random)\s*\(',
                severity=Severity.MEDIUM,
                description="Weak random number generation for security",
                cwe_id="CWE-338",
                context_multipliers={'token': 2.0, 'key': 2.5, 'nonce': 2.0}
            ),
            
            # Path Traversal - Enhanced
            VulnerabilityPattern(
                name="path_traversal_concat",
                pattern=r'open\s*\(\s*.*\+.*\)',
                severity=Severity.MEDIUM,
                description="Path traversal via string concatenation",
                cwe_id="CWE-22",
                context_multipliers={'user': 1.5, 'filename': 2.0}
            ),
            VulnerabilityPattern(
                name="file_access_user_input",
                pattern=r'open\s*\(\s*.*input.*\)',
                severity=Severity.HIGH,
                description="File access with user input",
                cwe_id="CWE-22"
            ),
            
            # Hardcoded Secrets - Enhanced detection
            VulnerabilityPattern(
                name="hardcoded_password",
                pattern=r'(password|passwd|pwd)\s*=\s*[\'"][^\'\"]{4,}[\'"]',
                severity=Severity.HIGH,
                description="Hardcoded password detected",
                cwe_id="CWE-798"
            ),
            VulnerabilityPattern(
                name="hardcoded_api_key",
                pattern=r'(api_key|apikey|key)\s*=\s*[\'"][A-Za-z0-9]{20,}[\'"]',
                severity=Severity.HIGH,
                description="Hardcoded API key detected",
                cwe_id="CWE-798"
            ),
            VulnerabilityPattern(
                name="hardcoded_secret",
                pattern=r'(secret|SECRET)\s*=\s*[\'"][^\'\"]{8,}[\'"]',
                severity=Severity.HIGH,
                description="Hardcoded secret detected",
                cwe_id="CWE-798"
            ),
            
            # Additional Enterprise Security Patterns
            VulnerabilityPattern(
                name="timing_attack_comparison",
                pattern=r'(password|token|hash)\s*==\s*',
                severity=Severity.MEDIUM,
                description="Timing attack vulnerability in comparison",
                cwe_id="CWE-208"
            ),
            VulnerabilityPattern(
                name="xml_external_entity",
                pattern=r'xml\.etree\.ElementTree\.parse',
                severity=Severity.HIGH,
                description="Potential XXE vulnerability",
                cwe_id="CWE-611"
            )
        ]
    
    def _initialize_enhanced_secret_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize enhanced secret patterns with production impact"""
        return {
            'aws_access_key': {
                'pattern': r'AKIA[0-9A-Z]{16}',
                'impact': 'CRITICAL',
                'description': 'AWS access key - full cloud infrastructure access'
            },
            'aws_secret_key': {
                'pattern': r'[0-9a-zA-Z/+]{40}',
                'impact': 'CRITICAL',
                'description': 'AWS secret key - infrastructure compromise'
            },
            'github_personal_token': {
                'pattern': r'ghp_[0-9a-zA-Z]{36}',
                'impact': 'HIGH',
                'description': 'GitHub personal access token'
            },
            'github_oauth_token': {
                'pattern': r'gho_[0-9a-zA-Z]{36}',
                'impact': 'HIGH',
                'description': 'GitHub OAuth token'
            },
            'openai_api_key': {
                'pattern': r'sk-[0-9a-zA-Z]{48}',
                'impact': 'HIGH',
                'description': 'OpenAI API key - potential cost and data exposure'
            },
            'anthropic_api_key': {
                'pattern': r'sk-ant-[0-9a-zA-Z\-]{10,}',
                'impact': 'HIGH',
                'description': 'Anthropic API key'
            },
            'stripe_secret_key': {
                'pattern': r'sk_live_[0-9a-zA-Z]{24}',
                'impact': 'CRITICAL',
                'description': 'Stripe live secret key - payment processing access'
            },
            'stripe_publishable_key': {
                'pattern': r'pk_live_[0-9a-zA-Z]{24}',
                'impact': 'MEDIUM',
                'description': 'Stripe publishable key'
            },
            'jwt_token': {
                'pattern': r'eyJ[0-9a-zA-Z_\-]+\.[0-9a-zA-Z_\-]+\.[0-9a-zA-Z_\-]+',
                'impact': 'HIGH',
                'description': 'JWT token - authentication bypass'
            },
            'private_key_header': {
                'pattern': r'-----BEGIN (RSA |DSA |EC |)PRIVATE KEY-----',
                'impact': 'CRITICAL',
                'description': 'Private key - cryptographic compromise'
            },
            'database_url': {
                'pattern': r'(postgresql|mysql|mongodb)://[^\\s]+',
                'impact': 'CRITICAL',
                'description': 'Database connection string with credentials'
            },
            'slack_token': {
                'pattern': r'xox[baprs]-[0-9a-zA-Z\-]+',
                'impact': 'MEDIUM',
                'description': 'Slack API token'
            },
            'generic_api_key_long': {
                'pattern': r'[\'"][a-zA-Z0-9]{32,}[\'"]',
                'impact': 'MEDIUM',
                'description': 'Potential API key or secret'
            }
        }
    
    def _initialize_context_keywords(self) -> Dict[str, List[str]]:
        """Initialize context keywords for severity escalation"""
        return {
            'authentication': ['auth', 'login', 'password', 'credential', 'token', 'session'],
            'payment': ['payment', 'billing', 'charge', 'transaction', 'money', 'stripe', 'paypal'],
            'user_data': ['user', 'profile', 'personal', 'private', 'sensitive'],
            'admin': ['admin', 'administrator', 'root', 'sudo', 'privilege'],
            'database': ['db', 'database', 'sql', 'query', 'table', 'schema'],
            'network': ['request', 'response', 'api', 'endpoint', 'url', 'http'],
            'file_system': ['file', 'path', 'directory', 'upload', 'download']
        }
    
    async def _analyze_implementation(self, code: str, context: Dict[str, Any]) -> AgentResult:
        """Enhanced security analysis with compound vulnerability detection"""
        issues = []
        metadata = {}
        
        # 1. Enhanced vulnerability pattern scanning
        vuln_issues = self._scan_enhanced_vulnerability_patterns(code)
        issues.extend(vuln_issues)
        
        # 2. Enhanced secrets detection
        secret_issues = self._detect_enhanced_secrets(code)
        issues.extend(secret_issues)
        
        # 3. Context-aware AST security analysis
        ast_security_issues = self._analyze_context_aware_security(code)
        issues.extend(ast_security_issues)
        
        # 4. Compound vulnerability detection
        compound_issues = self._detect_compound_vulnerabilities(issues, code)
        issues.extend(compound_issues)
        
        # 5. Enterprise security compliance check
        compliance_issues = self._check_enterprise_compliance(code, issues)
        issues.extend(compliance_issues)
        
        # 6. Dependency vulnerability checking (if enabled)
        if self.check_dependencies and REQUESTS_AVAILABLE:
            dep_issues = await self._check_dependency_vulnerabilities(code)
            issues.extend(dep_issues)
        
        # Calculate enhanced security metrics
        security_metrics = self._calculate_enhanced_security_metrics(issues, code)
        metadata['security_metrics'] = security_metrics.__dict__
        
        # Calculate enterprise security score
        overall_score = self._calculate_enterprise_security_score(issues, security_metrics)
        
        return AgentResult(
            agent_name=self.name,
            execution_time=0.0,
            overall_score=overall_score,
            issues=issues,
            metadata=metadata
        )
    
    def _scan_enhanced_vulnerability_patterns(self, code: str) -> List[VerificationIssue]:
        """Scan with context-aware severity escalation"""
        issues = []
        lines = code.splitlines()
        
        for pattern_def in self.vulnerability_patterns:
            pattern = re.compile(pattern_def.pattern, re.IGNORECASE | re.MULTILINE)
            
            for line_num, line in enumerate(lines, 1):
                matches = pattern.findall(line)
                if matches:
                    # Apply context-aware severity escalation
                    escalated_severity = self._escalate_severity_by_context(
                        pattern_def, line, code
                    )
                    
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_pattern_confidence(pattern_def, line)
                    
                    issues.append(VerificationIssue(
                        type=pattern_def.name,
                        severity=escalated_severity,
                        message=f"{pattern_def.description}",
                        line_number=line_num,
                        suggestion=self._generate_context_suggestion(pattern_def, line),
                        confidence=confidence
                    ))
        
        return issues
    
    def _escalate_severity_by_context(self, pattern: VulnerabilityPattern, 
                                     line: str, code: str) -> Severity:
        """Escalate severity based on code context"""
        base_severity = pattern.severity
        
        if not pattern.context_multipliers:
            return base_severity
        
        # Check for context keywords in line and surrounding context
        line_lower = line.lower()
        code_lower = code.lower()
        
        max_multiplier = 1.0
        for context_keyword, multiplier in pattern.context_multipliers.items():
            if context_keyword in line_lower or context_keyword in code_lower:
                max_multiplier = max(max_multiplier, multiplier)
        
        # Apply escalation based on multiplier
        if max_multiplier >= 2.5:
            if base_severity == Severity.HIGH:
                return Severity.CRITICAL
            elif base_severity == Severity.MEDIUM:
                return Severity.HIGH
        elif max_multiplier >= 1.5:
            if base_severity == Severity.MEDIUM:
                return Severity.HIGH
            elif base_severity == Severity.LOW:
                return Severity.MEDIUM
        
        return base_severity
    
    def _calculate_pattern_confidence(self, pattern: VulnerabilityPattern, line: str) -> float:
        """Calculate confidence based on pattern specificity and context"""
        base_confidence = 0.8
        
        # Increase confidence for specific patterns
        if pattern.cwe_id:
            base_confidence += 0.1
        
        # Increase confidence for context matches
        if pattern.context_multipliers:
            line_lower = line.lower()
            context_matches = sum(1 for keyword in pattern.context_multipliers.keys() 
                                if keyword in line_lower)
            base_confidence += context_matches * 0.05
        
        return min(1.0, base_confidence)
    
    def _generate_context_suggestion(self, pattern: VulnerabilityPattern, line: str) -> str:
        """Generate context-aware suggestions"""
        base_suggestion = f"Address {pattern.name}. CWE: {pattern.cwe_id}"
        
        # Add specific suggestions based on vulnerability type
        if 'sql' in pattern.name:
            return "Use parameterized queries or ORM to prevent SQL injection"
        elif 'command' in pattern.name:
            return "Use subprocess with shell=False and validate all inputs"
        elif 'eval' in pattern.name or 'exec' in pattern.name:
            return "Replace eval/exec with safer alternatives like ast.literal_eval"
        elif 'pickle' in pattern.name:
            return "Use secure serialization formats like JSON for untrusted data"
        elif 'md5' in pattern.name or 'sha1' in pattern.name:
            return "Use SHA-256 or stronger hash algorithms for security purposes"
        elif 'random' in pattern.name:
            return "Use secrets module for cryptographically secure random generation"
        else:
            return base_suggestion
    
    def _detect_enhanced_secrets(self, code: str) -> List[VerificationIssue]:
        """Enhanced secret detection with production impact assessment"""
        issues = []
        lines = code.splitlines()
        
        # Pattern-based secret detection with impact assessment
        for secret_type, secret_info in self.secret_patterns.items():
            pattern = secret_info['pattern']
            impact = secret_info['impact']
            description = secret_info['description']
            
            regex = re.compile(pattern)
            
            for line_num, line in enumerate(lines, 1):
                matches = regex.findall(line)
                for match in matches:
                    # Map impact to severity
                    severity = self._impact_to_severity(impact)
                    
                    issues.append(VerificationIssue(
                        type="secret",
                        severity=severity,
                        message=f"Hardcoded {secret_type}: {description}",
                        line_number=line_num,
                        suggestion=f"Move {secret_type} to environment variables or secure vault",
                        confidence=0.95
                    ))
        
        # Enhanced entropy-based detection
        entropy_secrets = self._detect_high_entropy_secrets(code)
        for secret in entropy_secrets:
            severity = self._impact_to_severity(secret.production_impact)
            
            issues.append(VerificationIssue(
                type="secret",
                severity=severity,
                message=f"High entropy string (entropy: {secret.entropy:.2f}) - potential {secret.type}",
                line_number=secret.line_number,
                suggestion="Review if this is a hardcoded secret that should be externalized",
                confidence=secret.confidence
            ))
        
        return issues
    
    def _impact_to_severity(self, impact: str) -> Severity:
        """Convert production impact to severity level"""
        impact_map = {
            'CRITICAL': Severity.CRITICAL,
            'HIGH': Severity.HIGH,
            'MEDIUM': Severity.MEDIUM,
            'LOW': Severity.LOW
        }
        return impact_map.get(impact, Severity.MEDIUM)
    
    def _detect_high_entropy_secrets(self, code: str) -> List[SecretCandidate]:
        """Enhanced entropy-based secret detection with impact assessment"""
        candidates = []
        lines = code.splitlines()
        
        # Enhanced patterns for different secret types
        enhanced_patterns = [
            (r'[\'"]([a-zA-Z0-9+/=]{40,})[\'"]', 'base64_key'),      # Base64-like
            (r'[\'"]([a-f0-9]{32,})[\'"]', 'hex_key'),              # Hex strings
            (r'[\'"]([a-zA-Z0-9]{32,})[\'"]', 'alphanumeric_key'),  # Random strings
            (r'sk-[a-zA-Z0-9]{20,}', 'api_key'),                   # API key format
            (r'[a-zA-Z0-9]{20,}-[a-zA-Z0-9]{20,}', 'composite_key') # Composite keys
        ]
        
        for line_num, line in enumerate(lines, 1):
            for pattern, key_type in enhanced_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    entropy = self._calculate_entropy(match)
                    
                    if entropy >= self.min_entropy_threshold:
                        # Assess production impact
                        impact = self._assess_production_impact(match, line, key_type)
                        confidence = self._calculate_secret_confidence(match, entropy, line)
                        
                        candidates.append(SecretCandidate(
                            value=match[:20] + "..." if len(match) > 20 else match,
                            type=key_type,
                            entropy=entropy,
                            line_number=line_num,
                            confidence=confidence,
                            production_impact=impact
                        ))
        
        return candidates
    
    def _assess_production_impact(self, secret: str, line: str, key_type: str) -> str:
        """Assess production impact of detected secret"""
        line_lower = line.lower()
        
        # Critical impact indicators
        if any(word in line_lower for word in ['prod', 'production', 'live', 'api_key']):
            return 'CRITICAL'
        
        # High impact indicators
        if any(word in line_lower for word in ['key', 'token', 'secret', 'password']):
            return 'HIGH'
        
        # Medium impact based on entropy and length
        if len(secret) > 40 and key_type in ['base64_key', 'api_key']:
            return 'MEDIUM'
        
        return 'LOW'
    
    def _calculate_secret_confidence(self, secret: str, entropy: float, line: str) -> float:
        """Calculate confidence that detected string is actually a secret"""
        base_confidence = min(0.9, (entropy - 3.0) / 5.0 + 0.4)
        
        # Increase confidence based on context
        line_lower = line.lower()
        if any(word in line_lower for word in ['key', 'token', 'secret', 'password', 'api']):
            base_confidence += 0.2
        
        # Increase confidence based on format
        if re.match(r'^[A-Za-z0-9+/]+=*$', secret):  # Base64-like
            base_confidence += 0.1
        elif re.match(r'^[a-f0-9]+$', secret):  # Hex
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy with enhanced accuracy"""
        if not text or len(text) < 8:
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
    
    def _analyze_context_aware_security(self, code: str) -> List[VerificationIssue]:
        """Context-aware AST security analysis"""
        issues = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues
        
        # Enhanced AST analysis with context awareness
        for node in ast.walk(tree):
            # Enhanced dangerous imports detection
            if isinstance(node, ast.Import):
                for alias in node.names:
                    risk_level = self._assess_import_risk(alias.name, code)
                    if risk_level > 0:
                        severity = Severity.HIGH if risk_level >= 2 else Severity.MEDIUM
                        issues.append(VerificationIssue(
                            type="dangerous_import",
                            severity=severity,
                            message=f"High-risk import: {alias.name} (risk level: {risk_level})",
                            line_number=getattr(node, 'lineno', None),
                            suggestion=f"Review security implications of {alias.name} usage"
                        ))
            
            # Enhanced function call analysis
            elif isinstance(node, ast.Call):
                call_issues = self._analyze_function_call_security(node, code)
                issues.extend(call_issues)
            
            # Enhanced assignment analysis
            elif isinstance(node, ast.Assign):
                assignment_issues = self._analyze_assignment_security(node)
                issues.extend(assignment_issues)
        
        return issues
    
    def _assess_import_risk(self, module_name: str, code: str) -> int:
        """Assess security risk level of imported modules"""
        high_risk_modules = {
            'os': 2, 'subprocess': 2, 'sys': 1, 'ctypes': 3,
            'importlib': 2, 'marshal': 3, 'pickle': 2, 'dill': 2,
            'eval': 3, 'exec': 3, 'compile': 2
        }
        
        base_risk = high_risk_modules.get(module_name, 0)
        
        # Increase risk if used with user input context
        if base_risk > 0:
            code_lower = code.lower()
            if any(word in code_lower for word in ['input', 'request', 'user', 'param']):
                base_risk += 1
        
        return base_risk
    
    def _analyze_function_call_security(self, node: ast.Call, code: str) -> List[VerificationIssue]:
        """Analyze function calls for security issues"""
        issues = []
        
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            # Critical function calls
            if func_name in ['eval', 'exec', 'compile']:
                issues.append(VerificationIssue(
                    type="code_execution",
                    severity=Severity.CRITICAL,
                    message=f"Code execution function: {func_name}()",
                    line_number=getattr(node, 'lineno', None),
                    suggestion=f"Replace {func_name}() with safer alternatives"
                ))
            
            # File operations with potential path traversal
            elif func_name == 'open' and len(node.args) > 0:
                if isinstance(node.args[0], ast.BinOp):
                    issues.append(VerificationIssue(
                        type="path_traversal",
                        severity=Severity.MEDIUM,
                        message="File path concatenation in open()",
                        line_number=getattr(node, 'lineno', None),
                            suggestion="Use pathlib or os.path.join() for safe path handling"
                    ))
        
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            
            # Dangerous method calls
            if attr_name in ['loads', 'load'] and hasattr(node.func, 'value'):
                if isinstance(node.func.value, ast.Name):
                    module_name = node.func.value.id
                    if module_name in ['pickle', 'marshal', 'dill']:
                        issues.append(VerificationIssue(
                            type="unsafe_deserialization",
                            severity=Severity.HIGH,
                            message=f"Unsafe deserialization: {module_name}.{attr_name}()",
                            line_number=getattr(node, 'lineno', None),
                            suggestion=f"Use secure alternatives to {module_name} for untrusted data"
                        ))
        
        return issues
    
    def _analyze_assignment_security(self, node: ast.Assign) -> List[VerificationIssue]:
        """Analyze assignments for security issues"""
        issues = []
        
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id.lower()
                
                # Check for credential-like variable names
                if any(keyword in var_name for keyword in 
                       ['password', 'secret', 'key', 'token', 'api_key', 'auth']):
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        # Assess severity based on string characteristics
                        secret_value = node.value.value
                        if len(secret_value) > 8:
                            issues.append(VerificationIssue(
                                type="hardcoded_secret",
                                severity=Severity.HIGH,
                                message=f"Hardcoded credential in variable: {target.id}",
                                line_number=getattr(node, 'lineno', None),
                                suggestion="Use environment variables or secure configuration"
                            ))
        
        return issues
    
    def _detect_compound_vulnerabilities(self, existing_issues: List[VerificationIssue], 
                                       code: str) -> List[VerificationIssue]:
        """Detect compound vulnerabilities with exponential risk"""
        compound_issues = []
        
        # Group issues by type for compound analysis
        issue_types = {}
        for issue in existing_issues:
            issue_type = issue.type
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(issue)
        
        # Define dangerous combinations with risk multipliers
        dangerous_combinations = [
            (['sql_injection_direct', 'hardcoded_secret'], 'Database + Credentials Exposure', 3.0),
            (['eval_execution', 'dangerous_import'], 'Code Execution + System Access', 2.5),
            (['secret', 'code_execution'], 'Credential Theft + Code Execution', 3.0),
            (['unsafe_deserialization', 'dangerous_import'], 'Deserialization + System Access', 2.0),
            (['path_traversal', 'file_access_user_input'], 'File System Compromise', 1.8),
            (['weak_random', 'hardcoded_secret'], 'Weak Crypto + Credential Exposure', 2.2),
            (['timing_attack_comparison', 'hardcoded_password'], 'Timing Attack + Weak Auth', 1.5)
        ]
        
        # Check for dangerous combinations
        for combo_types, description, risk_multiplier in dangerous_combinations:
            if all(any(issue_type in existing_type for existing_type in issue_types.keys()) 
                   for issue_type in combo_types):
                
                compound_issues.append(VerificationIssue(
                    type="compound_vulnerability",
                    severity=Severity.CRITICAL,
                    message=f"Compound vulnerability: {description} (risk×{risk_multiplier})",
                    suggestion=f"Address all components: {', '.join(combo_types)}",
                    confidence=0.95
                ))
        
        # Check for security issue cascades
        security_issue_count = len([i for i in existing_issues if i.severity in [Severity.HIGH, Severity.CRITICAL]])
        if security_issue_count >= 3:
            compound_issues.append(VerificationIssue(
                type="security_cascade",
                severity=Severity.CRITICAL,
                message=f"Security cascade: {security_issue_count} high/critical security issues",
                suggestion="Multiple security vulnerabilities create unacceptable production risk",
                confidence=1.0
            ))
        
        return compound_issues
    
    def _check_enterprise_compliance(self, code: str, issues: List[VerificationIssue]) -> List[VerificationIssue]:
        """Check enterprise security compliance standards"""
        compliance_issues = []
        
        # Count issues by severity
        critical_count = len([i for i in issues if i.severity == Severity.CRITICAL])
        high_count = len([i for i in issues if i.severity == Severity.HIGH])
        secret_count = len([i for i in issues if i.type == 'secret'])
        
        # Enterprise compliance checks
        if critical_count > self.enterprise_thresholds['max_critical_vulnerabilities']:
            compliance_issues.append(VerificationIssue(
                type="compliance_failure",
                severity=Severity.CRITICAL,
                message=f"Enterprise compliance failure: {critical_count} critical vulnerabilities exceed limit",
                suggestion="Zero critical vulnerabilities required for enterprise deployment"
            ))
        
        if high_count > self.enterprise_thresholds['max_high_vulnerabilities']:
            compliance_issues.append(VerificationIssue(
                type="compliance_failure",
                severity=Severity.HIGH,
                message=f"Enterprise compliance failure: {high_count} high vulnerabilities exceed limit",
                suggestion="Reduce high-severity vulnerabilities for enterprise compliance"
            ))
        
        if secret_count > self.enterprise_thresholds['max_secrets_per_file']:
            compliance_issues.append(VerificationIssue(
                type="compliance_failure",
                severity=Severity.HIGH,
                message=f"Enterprise compliance failure: {secret_count} hardcoded secrets detected",
                suggestion="Zero hardcoded secrets required for enterprise deployment"
            ))
        
        # Check for required crypto compliance
        if self.enterprise_thresholds['crypto_compliance_required']:
            crypto_compliance = self._check_crypto_compliance(code)
            if not crypto_compliance['compliant']:
                compliance_issues.append(VerificationIssue(
                    type="crypto_compliance_failure",
                    severity=Severity.HIGH,
                    message="Cryptographic compliance failure: " + crypto_compliance['reason'],
                    suggestion="Use enterprise-approved cryptographic algorithms"
                ))
        
        return compliance_issues
    
    def _check_crypto_compliance(self, code: str) -> Dict[str, Any]:
        """Check cryptographic compliance with enterprise standards"""
        # Approved algorithms for enterprise use
        approved_algorithms = {
            'hash': ['sha256', 'sha512', 'sha3_256', 'sha3_512'],
            'symmetric': ['aes', 'chacha20'],
            'random': ['secrets.token_bytes', 'secrets.token_hex', 'os.urandom']
        }
        
        # Check for deprecated/weak algorithms
        weak_patterns = [
            r'hashlib\.(md5|sha1)\s*\(',
            r'random\.(random|randint)\s*\(',
            r'DES|3DES|RC4'
        ]
        
        for pattern in weak_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return {
                    'compliant': False,
                    'reason': 'Weak cryptographic algorithms detected'
                }
        
        return {'compliant': True, 'reason': 'No weak crypto detected'}
    
    async def _check_dependency_vulnerabilities(self, code: str) -> List[VerificationIssue]:
        """Enhanced dependency vulnerability checking"""
        issues = []
        
        try:
            # Extract imports
            tree = ast.parse(code)
            imports = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
            
            # Enhanced known vulnerable packages
            known_vulnerable = {
                'pickle': {
                    'description': "Arbitrary code execution during deserialization",
                    'severity': Severity.HIGH,
                    'cve': 'CVE-2022-48560'
                },
                'marshal': {
                    'description': "Arbitrary code execution capabilities",
                    'severity': Severity.HIGH,
                    'cve': 'N/A'
                },
                'yaml': {
                    'description': "Code execution via yaml.load()",
                    'severity': Severity.MEDIUM,
                    'cve': 'CVE-2017-18342'
                },
                'dill': {
                    'description': "Arbitrary code execution during deserialization",
                    'severity': Severity.HIGH,
                    'cve': 'N/A'
                }
            }
            
            for imp in imports:
                if imp in known_vulnerable:
                    vuln_info = known_vulnerable[imp]
                    issues.append(VerificationIssue(
                        type="vulnerable_dependency",
                        severity=vuln_info['severity'],
                        message=f"Vulnerable dependency: {imp} - {vuln_info['description']}",
                        suggestion=f"Review {imp} usage or use safer alternatives. CVE: {vuln_info['cve']}"
                    ))
        
        except Exception:
            pass
        
        return issues
    
    def _calculate_enhanced_security_metrics(self, issues: List[VerificationIssue], 
                                           code: str) -> SecurityMetrics:
        """Calculate enhanced security metrics with enterprise context"""
        vuln_count = len([i for i in issues if 'injection' in i.type or 'execution' in i.type])
        secrets_count = len([i for i in issues if i.type == 'secret'])
        unsafe_patterns = len([i for i in issues if i.type in ['unsafe_deserialization', 'dangerous_import']])
        dep_issues = len([i for i in issues if i.type == 'vulnerable_dependency'])
        compound_vulns = len([i for i in issues if 'compound' in i.type])
        
        # Categorize issues with enterprise focus
        categories = {}
        for issue in issues:
            category = issue.type
            categories[category] = categories.get(category, 0) + 1
        
        # Enhanced risk scoring with compound vulnerability multipliers
        base_severity_scores = {
            Severity.LOW: 1,
            Severity.MEDIUM: 4,
            Severity.HIGH: 10,
            Severity.CRITICAL: 25
        }
        
        total_risk_score = 0
        for issue in issues:
            base_score = base_severity_scores.get(issue.severity, 1)
            
            # Apply multipliers for high-impact issue types
            if 'compound' in issue.type:
                base_score *= 2.0
            elif issue.type in ['code_execution', 'sql_injection_direct']:
                base_score *= 1.5
            elif issue.type == 'secret' and issue.severity == Severity.CRITICAL:
                base_score *= 1.3
            
            total_risk_score += base_score
        
        # Context-based risk assessment
        context_risks = self._assess_context_risks(code, issues)
        
        return SecurityMetrics(
            vulnerability_count=vuln_count,
            secrets_found=secrets_count,
            unsafe_patterns=unsafe_patterns,
            dependency_issues=dep_issues,
            total_risk_score=total_risk_score,
            compound_vulnerabilities=compound_vulns,
            categories=categories,
            context_risks=context_risks
        )
    
    def _assess_context_risks(self, code: str, issues: List[VerificationIssue]) -> Dict[str, float]:
        """Assess security risks based on code context"""
        context_risks = {
            'authentication_risk': 0.0,
            'data_exposure_risk': 0.0,
            'system_compromise_risk': 0.0,
            'compliance_risk': 0.0
        }
        
        code_lower = code.lower()
        
        # Authentication risk
        auth_keywords = ['auth', 'login', 'password', 'credential', 'session']
        auth_issues = [i for i in issues if any(keyword in i.message.lower() for keyword in auth_keywords)]
        if auth_issues:
            context_risks['authentication_risk'] = min(1.0, len(auth_issues) / 3.0)
        
        # Data exposure risk
        data_keywords = ['database', 'user', 'personal', 'private', 'secret']
        data_issues = [i for i in issues if any(keyword in i.message.lower() for keyword in data_keywords)]
        if data_issues:
            context_risks['data_exposure_risk'] = min(1.0, len(data_issues) / 4.0)
        
        # System compromise risk
        system_issues = [i for i in issues if i.type in ['code_execution', 'command_injection', 'dangerous_import']]
        if system_issues:
            context_risks['system_compromise_risk'] = min(1.0, len(system_issues) / 2.0)
        
        # Compliance risk
        compliance_issues = [i for i in issues if 'compliance' in i.type]
        if compliance_issues:
            context_risks['compliance_risk'] = 1.0
        
        return context_risks
    
    def _calculate_enterprise_security_score(self, issues: List[VerificationIssue], 
                                           metrics: SecurityMetrics) -> float:
        """Calculate enterprise security score with zero-tolerance for critical issues"""
        if not issues:
            return 1.0
        
        # Enterprise security scoring with zero tolerance
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        high_issues = [i for i in issues if i.severity == Severity.HIGH]
        compound_issues = [i for i in issues if 'compound' in i.type]
        
        # Zero tolerance for critical security issues
        if critical_issues or compound_issues:
            return max(0.1, 0.4 - len(critical_issues) * 0.15)
        
        # Very aggressive penalties for high-severity security issues
        if len(high_issues) >= 2:
            return max(0.2, 0.5 - len(high_issues) * 0.1)
        elif len(high_issues) >= 1:
            return max(0.4, 0.7 - len(high_issues) * 0.15)
        
        # Standard scoring for medium/low issues
        medium_issues = [i for i in issues if i.severity == Severity.MEDIUM]
        low_issues = [i for i in issues if i.severity == Severity.LOW]
        
        # Calculate penalty based on enterprise standards
        total_penalty = (
            len(medium_issues) * 0.08 +  # 8% penalty per medium issue
            len(low_issues) * 0.03       # 3% penalty per low issue
        )
        
        # Apply context-based additional penalties
        context_penalty = 0.0
        for risk_type, risk_value in metrics.context_risks.items():
            if risk_value > 0.5:  # High context risk
                context_penalty += risk_value * 0.1
        
        final_score = max(0.0, 1.0 - total_penalty - context_penalty)
        return round(final_score, 3)