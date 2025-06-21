# tests/integration/test_orchestrator.py
"""
Integration tests for the AsyncOrchestrator and complete verification pipeline.
Tests end-to-end functionality, parallel execution, caching, and error handling.
"""

import pytest
import asyncio
import time
import tempfile
import os
from pathlib import Path

# Import the orchestration components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.orchestration.async_orchestrator import (
    AsyncOrchestrator, VerificationConfig, VerificationReport
)
from src.orchestration.result_aggregator import ResultAggregator
from src.orchestration.caching_layer import CachingLayer
from src.agents.base_agent import Severity


@pytest.fixture
def orchestrator():
    """Create orchestrator for testing"""
    config = VerificationConfig.default()
    config.max_execution_time = 10.0  # Shorter timeout for tests
    return AsyncOrchestrator(config)


@pytest.fixture
def simple_code():
    """Simple test code"""
    return '''
def add_numbers(a, b):
    """Add two numbers together."""
    return a + b

def multiply(x, y):
    return x * y
'''


@pytest.fixture
def problematic_code():
    """Code with various issues for testing"""
    return '''
import os

# Security issue - hardcoded secret
api_key = "sk-1234567890abcdef"

def unsafe_command(user_input):
    # Command injection vulnerability
    os.system("echo " + user_input)

def bad_style_function(x,y,z):
    if x>0:
     if y>0:
      if z>0:
       return x+y+z
      else:
       return None
     else:
      return None
    else:
     return None
'''


class TestAsyncOrchestrator:
    """Test the main orchestration engine"""
    
    @pytest.mark.asyncio
    async def test_basic_verification(self, orchestrator, simple_code):
        """Test basic code verification"""
        report = await orchestrator.verify_code(simple_code)
        
        assert isinstance(report, VerificationReport)
        assert report.overall_score >= 0.0
        assert report.overall_score <= 1.0
        assert report.execution_time > 0
        assert len(report.agent_results) > 0
        assert report.overall_status in ["PASS", "FAIL", "WARNING"]
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, orchestrator, simple_code):
        """Test that agents execute in parallel"""
        start_time = time.time()
        report = await orchestrator.verify_code(simple_code)
        total_time = time.time() - start_time
        
        # Sum of individual agent times should be greater than total time
        # (indicating parallel execution)
        agent_time_sum = sum(result.execution_time for result in report.agent_results.values())
        
        # Allow some overhead but parallel should be faster
        assert total_time < agent_time_sum * 0.8 or total_time < 1.0  # Quick execution might not show parallelism
    
    @pytest.mark.asyncio
    async def test_problematic_code_detection(self, orchestrator, problematic_code):
        """Test detection of various code issues"""
        report = await orchestrator.verify_code(problematic_code)
        
        # Should detect issues
        assert len(report.aggregated_issues) > 0
        
        # Should have security issues
        security_issues = [i for i in report.aggregated_issues if 'secret' in i.type or 'security' in i.type]
        assert len(security_issues) > 0
        
        # Should have style issues
        style_issues = [i for i in report.aggregated_issues if 'style' in i.type or 'formatting' in i.type]
        assert len(style_issues) > 0
        
        # Overall score should be lower
        assert report.overall_score < 0.8
    
    @pytest.mark.asyncio
    async def test_agent_failure_handling(self, orchestrator):
        """Test handling of agent failures"""
        # Code that might cause parsing issues
        broken_code = "def incomplete_function(:"
        
        report = await orchestrator.verify_code(broken_code)
        
        # Should still return a report
        assert isinstance(report, VerificationReport)
        
        # Some agents might fail but should be handled gracefully
        failed_agents = [name for name, result in report.agent_results.items() if not result.success]
        
        # Even with failures, should get some kind of score
        assert report.overall_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_batch_verification(self, orchestrator, simple_code, problematic_code):
        """Test batch verification of multiple code samples"""
        samples = [
            {"code": simple_code, "context": {"name": "simple"}},
            {"code": problematic_code, "context": {"name": "problematic"}}
        ]
        
        reports = await orchestrator.batch_verify(samples)
        
        assert len(reports) == 2
        assert all(isinstance(report, VerificationReport) for report in reports)
        
        # Simple code should score higher than problematic code
        assert reports[0].overall_score > reports[1].overall_score
    
    @pytest.mark.asyncio
    async def test_health_check(self, orchestrator):
        """Test system health check"""
        health = await orchestrator.health_check()
        
        assert 'overall_status' in health
        assert 'agents' in health
        assert health['overall_status'] in ['healthy', 'degraded', 'unhealthy']
        
        # Should check all agents
        assert len(health['agents']) > 0
    
    @pytest.mark.asyncio
    async def test_execution_stats(self, orchestrator, simple_code):
        """Test execution statistics tracking"""
        # Run a few verifications
        for _ in range(3):
            await orchestrator.verify_code(simple_code)
        
        stats = orchestrator.get_execution_stats()
        
        assert stats['total_verifications'] == 3
        assert stats['avg_execution_time'] > 0
        assert 'agent_performance' in stats
    
    @pytest.mark.asyncio
    async def test_configuration_loading(self):
        """Test configuration loading and customization"""
        # Test default configuration
        config = VerificationConfig.default()
        assert 'correctness' in config.enabled_agents
        assert config.parallel_execution is True
        
        # Test custom configuration
        custom_config = VerificationConfig(
            enabled_agents={'security', 'style'},
            agent_configs={},
            thresholds={'overall_min_score': 0.9}
        )
        
        orchestrator = AsyncOrchestrator(custom_config)
        report = await orchestrator.verify_code("def test(): pass")
        
        # Should only run enabled agents
        assert len(report.agent_results) == 2
        assert 'security' in report.agent_results
        assert 'style' in report.agent_results


class TestResultAggregator:
    """Test result aggregation functionality"""
    
    def test_issue_clustering(self):
        """Test issue deduplication and clustering"""
        from src.agents.base_agent import VerificationIssue, AgentResult
        
        config = VerificationConfig.default()
        aggregator = ResultAggregator(config)
        
        # Create duplicate issues
        issues = [
            VerificationIssue("style", Severity.LOW, "Line too long", line_number=1),
            VerificationIssue("style", Severity.LOW, "Line too long", line_number=1),
            VerificationIssue("security", Severity.HIGH, "SQL injection", line_number=5)
        ]
        
        clustered = aggregator._cluster_similar_issues(issues)
        
        # Should deduplicate similar issues
        assert len(clustered) < len(issues)
    
    def test_unified_scoring(self):
        """Test unified score calculation"""
        from src.agents.base_agent import AgentResult
        
        config = VerificationConfig.default()
        aggregator = ResultAggregator(config)
        
        agent_results = {
            'correctness': AgentResult("correctness", 0.1, 0.9, [], {}, True),
            'security': AgentResult("security", 0.1, 0.8, [], {}, True),
            'performance': AgentResult("performance", 0.1, 0.7, [], {}, True),
            'style': AgentResult("style", 0.1, 0.6, [], {}, True)
        }
        
        score = aggregator._calculate_unified_score(agent_results)
        
        # Should be weighted average
        assert 0.0 <= score <= 1.0
        assert score > 0.7  # Should be reasonable given input scores
    
    def test_status_determination(self):
        """Test status determination logic"""
        from src.agents.base_agent import VerificationIssue
        
        config = VerificationConfig.default()
        aggregator = ResultAggregator(config)
        
        # Test with critical issues
        critical_issues = [VerificationIssue("security", Severity.CRITICAL, "Critical issue")]
        status = aggregator._determine_status(0.9, critical_issues)
        assert status == "FAIL"
        
        # Test with high score and no critical issues
        no_issues = []
        status = aggregator._determine_status(0.95, no_issues)
        assert status == "PASS"
        
        # Test with medium score
        medium_issues = [VerificationIssue("style", Severity.MEDIUM, "Medium issue")]
        status = aggregator._determine_status(0.75, medium_issues)
        assert status in ["WARNING", "PASS"]


class TestCachingLayer:
    """Test caching functionality"""
    
    @pytest.mark.asyncio
    async def test_basic_caching(self):
        """Test basic cache operations"""
        cache = CachingLayer(enabled=True, ttl=60)
        
        # Test cache miss
        result = await cache.get("test_key")
        assert result is None
        
        # Test cache set and hit
        test_value = {"data": "test"}
        success = await cache.set("test_key", test_value)
        assert success
        
        cached_result = await cache.get("test_key")
        assert cached_result == test_value
        
        await cache.cleanup()
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test cache TTL and expiration"""
        cache = CachingLayer(enabled=True, ttl=1)  # 1 second TTL
        
        await cache.set("expire_test", "value", ttl=1)
        
        # Should be available immediately
        result = await cache.get("expire_test")
        assert result == "value"
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired
        result = await cache.get("expire_test")
        assert result is None
        
        await cache.cleanup()
    
    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test cache statistics tracking"""
        cache = CachingLayer(enabled=True)
        
        # Generate some cache activity
        await cache.set("key1", "value1")
        await cache.get("key1")  # Hit
        await cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        
        assert stats.total_requests >= 2
        assert stats.cache_hits >= 1
        assert stats.cache_misses >= 1
        assert 0.0 <= stats.hit_rate <= 1.0
        
        await cache.cleanup()
    
    @pytest.mark.asyncio
    async def test_cache_disabled(self):
        """Test cache behavior when disabled"""
        cache = CachingLayer(enabled=False)
        
        # Should always return None when disabled
        await cache.set("test", "value")
        result = await cache.get("test")
        assert result is None
        
        await cache.cleanup()


class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_excellent_code(self):
        """Test complete pipeline with high-quality code"""
        excellent_code = '''
def calculate_factorial(n: int) -> int:
    """
    Calculate factorial of a non-negative integer.
    
    Args:
        n: Non-negative integer
        
    Returns:
        Factorial of n
        
    Raises:
        ValueError: If n is negative
        TypeError: If n is not an integer
    """
    if not isinstance(n, int):
        raise TypeError("Input must be an integer")
    if n < 0:
        raise ValueError("Input must be non-negative")
    
    result = 1
    for i in range(1, n + 1):
        result *= i
    
    return result
'''
        
        orchestrator = AsyncOrchestrator()
        report = await orchestrator.verify_code(excellent_code)
        
        # Should get high scores for excellent code
        assert report.overall_score > 0.8
        assert report.overall_status in ["PASS", "WARNING"]
        
        # Should have few or no issues
        critical_issues = [i for i in report.aggregated_issues if i.severity == Severity.CRITICAL]
        assert len(critical_issues) == 0
        
        await orchestrator.cleanup()
    
    @pytest.mark.asyncio
    async def test_full_pipeline_poor_code(self):
        """Test complete pipeline with poor-quality code"""
        poor_code = '''
# No docstring, security issues, poor style
api_secret="sk-abcdef123456"
def bad_func(x,y,z):
 result=eval(x)
 os.system("rm " + y)
 for i in range(1000):
  for j in range(1000):
   result+=i*j
 return result
'''
        
        orchestrator = AsyncOrchestrator()
        report = await orchestrator.verify_code(poor_code)
        
        # Should detect multiple serious issues
        assert report.overall_score < 0.5
        assert report.overall_status == "FAIL"
        
        # Should find security issues
        security_issues = [i for i in report.aggregated_issues 
                          if i.severity in [Severity.CRITICAL, Severity.HIGH]]
        assert len(security_issues) > 0
        
        await orchestrator.cleanup()
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self):
        """Test that system meets performance benchmarks"""
        test_code = "def hello(): return 'world'"
        orchestrator = AsyncOrchestrator()
        
        # Test single verification speed
        start_time = time.time()
        report = await orchestrator.verify_code(test_code)
        execution_time = time.time() - start_time
        
        # Should complete quickly (under 5 seconds for simple code)
        assert execution_time < 5.0
        assert report.execution_time < 5.0
        
        # Test batch verification
        samples = [{"code": test_code} for _ in range(5)]
        
        start_time = time.time()
        reports = await orchestrator.batch_verify(samples)
        batch_time = time.time() - start_time
        
        # Batch should be efficient
        assert len(reports) == 5
        assert batch_time < 10.0  # Should handle 5 samples quickly
        
        await orchestrator.cleanup()


if __name__ == "__main__":
    # Run tests manually if needed
    import pytest
    pytest.main([__file__, "-v"])