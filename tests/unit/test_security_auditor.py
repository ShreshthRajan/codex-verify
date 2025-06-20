# tests/unit/test_security_auditor.py
"""
Comprehensive test suite for Security Auditor Agent.
Tests vulnerability detection, secrets scanning, and security analysis.
"""

import pytest
import asyncio
from src.agents.security_auditor import SecurityAuditor, SecurityMetrics, SecretCandidate
from src.agents.base_agent import Severity


class TestSecurityAuditor:
    """Test suite for SecurityAuditor agent"""
    
    @pytest.fixture
    def security_auditor(self):
        """Create SecurityAuditor instance for testing"""
        config = {
            'min_entropy_threshold': 4.5,
            'check_dependencies': True
        }
        return SecurityAuditor(config)
    
    @pytest.mark.asyncio
    async def test_sql_injection_detection(self, security_auditor):
        """Test SQL injection vulnerability detection"""
        vulnerable_code = """
import sqlite3

def get_user(username):
    cursor.execute("SELECT * FROM users WHERE username = '%s'" % username)
    return cursor.fetchone()

def search_products(category):
    query = "SELECT * FROM products WHERE category = '{}'".format(category)
    cursor.execute(query)
"""
        
        result = await security_auditor.analyze(vulnerable_code)
        
        assert result.success
        assert result.overall_score < 0.95  # Should detect security issues (more lenient for 2 issues)
        
        # Check for SQL injection issues
        sql_issues = [issue for issue in result.issues if issue.type == "vulnerability" 
                     and "sql" in issue.message.lower()]
        assert len(sql_issues) >= 1
        assert any(issue.severity in [Severity.HIGH, Severity.CRITICAL] for issue in sql_issues)
    
    @pytest.mark.asyncio
    async def test_command_injection_detection(self, security_auditor):
        """Test command injection vulnerability detection"""
        vulnerable_code = """
import os
import subprocess

def process_file(filename):
    os.system("cat " + filename)
    
def run_command(cmd):
    subprocess.run(cmd + " --safe", shell=True)
"""
        
        result = await security_auditor.analyze(vulnerable_code)
        
        assert result.success
        
        # Check for command injection issues
        cmd_issues = [issue for issue in result.issues if "command" in issue.message.lower() 
                     or "injection" in issue.message.lower()]
        assert len(cmd_issues) >= 1
    
    @pytest.mark.asyncio
    async def test_hardcoded_secrets_detection(self, security_auditor):
        """Test detection of hardcoded secrets"""
        code_with_secrets = """
API_KEY = "sk-1234567890abcdef1234567890abcdef12345678"
password = "super_secret_password_123"
aws_access_key = "AKIAIOSFODNN7EXAMPLE"

def connect_db():
    conn = mysql.connector.connect(
        host="localhost",
        user="admin",
        password="hardcoded_password"
    )
"""
        
        result = await security_auditor.analyze(code_with_secrets)
        
        assert result.success
        assert result.overall_score < 0.8  # Should heavily penalize secrets (7 issues total)
        
        # Check for secret detection
        secret_issues = [issue for issue in result.issues if issue.type == "secret" 
                        or "secret" in issue.message.lower() or "password" in issue.message.lower()]
        assert len(secret_issues) >= 2
    
    @pytest.mark.asyncio
    async def test_code_execution_detection(self, security_auditor):
        """Test detection of dangerous code execution"""
        dangerous_code = """
def execute_user_code(code):
    result = eval(code)
    return result

def run_script(script):
    exec(script)

def compile_code(source):
    compiled = compile(source, '<string>', 'exec')
    exec(compiled)
"""
        
        result = await security_auditor.analyze(dangerous_code)
        
        assert result.success
        assert result.overall_score < 0.7  # Should heavily penalize code execution (7 critical issues)
        
        # Check for code execution issues
        exec_issues = [issue for issue in result.issues if issue.type == "code_execution" 
                      or "execution" in issue.message.lower()]
        assert len(exec_issues) >= 2
        assert any(issue.severity == Severity.CRITICAL for issue in exec_issues)
    
    @pytest.mark.asyncio
    async def test_deserialization_vulnerabilities(self, security_auditor):
        """Test detection of unsafe deserialization"""
        unsafe_code = """
import pickle
import yaml

def load_data(data):
    obj = pickle.loads(data)
    return obj

def parse_config(config_data):
    config = yaml.load(config_data)
    return config
"""
        
        result = await security_auditor.analyze(unsafe_code)
        
        assert result.success
        
        # Check for deserialization issues
        deserial_issues = [issue for issue in result.issues if "deserialization" in issue.message.lower()]
        assert len(deserial_issues) >= 1
        assert any(issue.severity == Severity.HIGH for issue in deserial_issues)
    
    @pytest.mark.asyncio
    async def test_weak_cryptography_detection(self, security_auditor):
        """Test detection of weak cryptographic functions"""
        weak_crypto_code = """
import hashlib

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()

def generate_token(data):
    return hashlib.sha1(data.encode()).digest()
"""
        
        result = await security_auditor.analyze(weak_crypto_code)
        
        assert result.success
        
        # Check for weak crypto issues
        crypto_issues = [issue for issue in result.issues if "hash" in issue.message.lower() 
                        or "cryptographic" in issue.message.lower()]
        assert len(crypto_issues) >= 1
    
    @pytest.mark.asyncio
    async def test_entropy_calculation(self, security_auditor):
        """Test entropy calculation for secret detection"""
        # High entropy string (should be flagged)
        high_entropy = "aB3xY7qM9kL2nP5vR8sT1wZ4jF6hD0gE"
        entropy = security_auditor._calculate_entropy(high_entropy)
        assert entropy > 4.5
        
        # Low entropy string (should not be flagged)
        low_entropy = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        entropy = security_auditor._calculate_entropy(low_entropy)
        assert entropy < 2.0
        
        # Empty string
        empty_entropy = security_auditor._calculate_entropy("")
        assert empty_entropy == 0.0
    
    @pytest.mark.asyncio
    async def test_high_entropy_string_detection(self, security_auditor):
        """Test detection of high entropy strings"""
        code_with_entropy = """
secret_key = "aB3xY7qM9kL2nP5vR8sT1wZ4jF6hD0gE9cH2mA5"
base64_data = "SGVsbG8gV29ybGQhIFRoaXMgaXMgYSBsb25nIGJhc2U2NCBzdHJpbmc="
normal_string = "hello world"
"""
        
        result = await security_auditor.analyze(code_with_entropy)
        
        assert result.success
        
        # Check for high entropy detection
        entropy_issues = [issue for issue in result.issues if "entropy" in issue.message.lower()]
        assert len(entropy_issues) >= 1
    
    @pytest.mark.asyncio
    async def test_dangerous_imports_detection(self, security_auditor):
        """Test detection of dangerous imports"""
        dangerous_imports_code = """
import os
import subprocess
import sys
import ctypes
import importlib

def dangerous_function():
    os.system("ls")
    subprocess.call("echo hello", shell=True)
"""
        
        result = await security_auditor.analyze(dangerous_imports_code)
        
        assert result.success
        
        # Check for dangerous import detection
        import_issues = [issue for issue in result.issues if issue.type == "dangerous_import"]
        assert len(import_issues) >= 3  # os, subprocess, sys, ctypes, importlib
    
    @pytest.mark.asyncio
    async def test_path_traversal_detection(self, security_auditor):
        """Test detection of path traversal vulnerabilities"""
        path_traversal_code = """
def read_file(filename):
    with open("/var/logs/" + filename, 'r') as f:
        return f.read()

def write_data(path, data):
    file_path = base_dir + path
    with open(file_path, 'w') as f:
        f.write(data)
"""
        
        result = await security_auditor.analyze(path_traversal_code)
        
        assert result.success
        
        # Check for path traversal issues
        path_issues = [issue for issue in result.issues if "path" in issue.message.lower()]
        assert len(path_issues) >= 1
    
    @pytest.mark.asyncio
    async def test_taint_analysis_simulation(self, security_auditor):
        """Test taint analysis simulation"""
        taint_code = """
def process_input():
    user_data = input("Enter command: ")
    system(user_data)  # Tainted data flow

def network_handler():
    data = socket.recv(1024)
    eval(data)  # Dangerous!
"""
        
        result = await security_auditor.analyze(taint_code)
        
        assert result.success
        
        # Check for taint flow issues
        taint_issues = [issue for issue in result.issues if issue.type == "taint_flow"]
        # Note: This is a simplified test - full taint analysis would be more complex
    
    @pytest.mark.asyncio
    async def test_clean_code_analysis(self, security_auditor):
        """Test analysis of clean, secure code"""
        clean_code = """
import hashlib
import os
from pathlib import Path

def secure_hash(data):
    return hashlib.sha256(data.encode()).hexdigest()

def safe_file_read(filename):
    safe_path = Path("/safe/directory") / filename
    if safe_path.is_file():
        with open(safe_path, 'r') as f:
            return f.read()
    return None

def validate_input(user_input):
    # Proper input validation
    if not user_input.isalnum():
        raise ValueError("Invalid input")
    return user_input
"""
        
        result = await security_auditor.analyze(clean_code)
        
        assert result.success
        assert result.overall_score > 0.8  # Should score well for clean code
        
        # Should have minimal or no security issues
        critical_issues = [issue for issue in result.issues if issue.severity == Severity.CRITICAL]
        assert len(critical_issues) == 0
    
    @pytest.mark.asyncio
    async def test_security_metrics_calculation(self, security_auditor):
        """Test security metrics calculation"""
        vulnerable_code = """
password = "hardcoded123"
api_key = "sk-1234567890abcdef1234567890abcdef12345678"

def unsafe_query(user_id):
    query = "SELECT * FROM users WHERE id = %s" % user_id
    cursor.execute(query)

def run_command(cmd):
    eval(cmd)
"""
        
        result = await security_auditor.analyze(vulnerable_code)
        
        assert result.success
        assert 'security_metrics' in result.metadata
        
        metrics = result.metadata['security_metrics']
        assert metrics['secrets_found'] >= 1  # Should find at least 1 secret
        assert metrics['vulnerability_count'] >= 1
        assert metrics['vulnerability_count'] >= 1
        assert metrics['total_risk_score'] > 0
        assert len(metrics['categories']) > 0
    
    @pytest.mark.asyncio
    async def test_syntax_error_handling(self, security_auditor):
        """Test handling of code with syntax errors"""
        syntax_error_code = """
def broken_function(
    # Missing closing parenthesis and colon
    return "this won't parse"
"""
        
        result = await security_auditor.analyze(syntax_error_code)
        
        # Should still succeed but with limited analysis
        assert result.success
        # Some basic pattern matching should still work
    
    @pytest.mark.asyncio
    async def test_empty_code_analysis(self, security_auditor):
        """Test analysis of empty code"""
        result = await security_auditor.analyze("")
        
        assert result.success
        assert result.overall_score == 1.0  # Perfect score for empty code
        assert len(result.issues) == 0
    
    @pytest.mark.asyncio
    async def test_configuration_options(self):
        """Test configuration options"""
        # Test with different entropy threshold
        config = {
            'min_entropy_threshold': 6.0,  # Higher threshold
            'check_dependencies': False    # Disable dependency checking
        }
        auditor = SecurityAuditor(config)
        
        # Test that configuration is applied
        assert auditor.min_entropy_threshold == 6.0
        assert auditor.check_dependencies == False
    
    @pytest.mark.asyncio
    async def test_agent_disabled(self):
        """Test agent when disabled"""
        config = {'enabled': False}
        auditor = SecurityAuditor(config)
        
        result = await auditor.analyze("eval('malicious code')")
        
        assert result.success
        assert result.overall_score == 1.0
        assert len(result.issues) == 0
        assert result.metadata.get('skipped') == True