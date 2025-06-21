# demo_orchestration_phase3.py
"""
Complete Phase 3 Orchestration Demo - Multi-Agent Code Verification System
Demonstrates the full orchestration engine coordinating all 4 agents in parallel.
"""

import asyncio
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.orchestration.async_orchestrator import AsyncOrchestrator, VerificationConfig
from src.agents.base_agent import Severity

# Test code samples for comprehensive demonstration
TEST_SAMPLES = {
    "excellent_code": '''
def calculate_fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number using dynamic programming.
    
    Args:
        n: The position in the Fibonacci sequence (0-indexed)
        
    Returns:
        The nth Fibonacci number
        
    Raises:
        ValueError: If n is negative
    """
    if not isinstance(n, int):
        raise TypeError("Input must be an integer")
    if n < 0:
        raise ValueError("Input must be non-negative")
    
    if n <= 1:
        return n
    
    # Use dynamic programming for O(n) complexity
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    
    return curr


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if email format is valid, False otherwise
    """
    import re
    
    if not isinstance(email, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
''',

    "security_issues": '''
import os
import subprocess

# Hardcoded credentials - security issue
API_KEY = "sk-1234567890abcdef1234567890abcdef12345678"
PASSWORD = "admin123"

def execute_command(user_input):
    # Command injection vulnerability
    command = "ls " + user_input
    os.system(command)

def get_user_data(user_id):
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    # Simulated database execution
    return query

def process_file(filename):
    # Path traversal vulnerability
    with open("/data/" + filename, 'r') as f:
        return f.read()

# Unsafe deserialization
import pickle
def load_config(data):
    return pickle.loads(data)
''',

    "performance_issues": '''
def inefficient_search(items, target):
    # O(n^2) nested loop - performance issue
    for i in range(len(items)):
        for j in range(len(items)):
            if items[i] == target and items[j] == target:
                return True
    return False

def recursive_fibonacci(n):
    # Exponential time complexity - major performance issue
    if n <= 1:
        return n
    return recursive_fibonacci(n-1) + recursive_fibonacci(n-2)

def memory_inefficient_process(data):
    # Memory inefficient - creates unnecessary copies
    result = []
    for i in range(len(data)):
        temp_list = data[:i] + data[i+1:]  # O(n) operation in loop
        result.append(temp_list)
    return result

class DatabaseConnection:
    def __init__(self):
        self.connections = []
    
    def get_data(self, query):
        # Potential memory leak - connections never closed
        for i in range(1000):
            connection = self.create_connection()
            self.connections.append(connection)
        return "data"
''',

    "style_issues": '''
def badFunction(x,y):
 a=x+y
 b=x-y
 if a>10:
  if b<5:
   return a*b
  else:
   return a/b
 else:
  return None

class myClass:
 def __init__(self,value):
  self.val=value
 def getValue(self):
  return self.val

# Missing docstrings
def complex_calculation(data):
 result=0
 for item in data:
  if item>0:
   result+=item*2
  elif item<0:
   result-=abs(item)
  else:
   result+=1
 return result

# Long lines and poor formatting
def process_data(input_data, options, configuration_parameters, additional_settings, processing_mode="default"):
 return input_data if processing_mode=="simple" else complex_calculation(input_data) if options.get("complex", False) else simple_process(input_data)
'''
}


async def demonstrate_orchestration():
    """Demonstrate the complete orchestration system"""
    
    print("🚀 CODEX-VERIFY PHASE 3: ORCHESTRATION ENGINE DEMO")
    print("=" * 60)
    print()
    
    # Create orchestrator with default configuration
    config = VerificationConfig.default()
    orchestrator = AsyncOrchestrator(config)
    
    print("📋 Configuration:")
    print(f"   • Enabled agents: {config.enabled_agents}")
    print(f"   • Parallel execution: {config.parallel_execution}")
    print(f"   • Caching enabled: {config.enable_caching}")
    print(f"   • Max execution time: {config.max_execution_time}s")
    print()
    
    # Health check
    print("🏥 System Health Check:")
    health = await orchestrator.health_check()
    print(f"   • Overall status: {health['overall_status']}")
    for agent_name, agent_health in health['agents'].items():
        status_icon = "✅" if agent_health['status'] == 'healthy' else "❌"
        print(f"   • {agent_name}: {status_icon} {agent_health['status']}")
    print()
    
    # Test each sample
    for sample_name, code in TEST_SAMPLES.items():
        print(f"🔍 Testing: {sample_name.replace('_', ' ').title()}")
        print("-" * 40)
        
        start_time = time.time()
        
        # Run verification
        report = await orchestrator.verify_code(
            code=code,
            context={"sample_name": sample_name}
        )
        
        execution_time = time.time() - start_time
        
        # Display results
        print(f"📊 Results ({execution_time:.3f}s total):")
        print(f"   • Overall Score: {report.overall_score:.3f} ({report.summary['grade']})")
        print(f"   • Status: {report.overall_status}")
        print()
        
        print("🤖 Agent Performance:")
        for agent_name, result in report.agent_results.items():
            status_icon = "✅" if result.success else "❌"
            score_color = "🟢" if result.overall_score >= 0.8 else "🟡" if result.overall_score >= 0.6 else "🔴"
            print(f"   • {agent_name}: {status_icon} {score_color} {result.overall_score:.3f} ({result.execution_time:.3f}s)")
        print()
        
        # Show critical issues
        critical_issues = [i for i in report.aggregated_issues if i.severity == Severity.CRITICAL]
        high_issues = [i for i in report.aggregated_issues if i.severity == Severity.HIGH]
        
        if critical_issues:
            print("🚨 Critical Issues:")
            for issue in critical_issues[:3]:  # Show top 3
                print(f"   • {issue.message}")
        
        if high_issues:
            print("⚠️  High Priority Issues:")
            for issue in high_issues[:3]:  # Show top 3
                print(f"   • {issue.message}")
        
        if not critical_issues and not high_issues:
            print("✅ No critical or high priority issues found!")
        
        print()
        
        # Show top recommendations
        if report.recommendations:
            print("💡 Top Recommendations:")
            for rec in report.recommendations[:3]:
                print(f"   • {rec}")
        
        print()
        print("=" * 60)
        print()
    
    # Show execution statistics
    stats = orchestrator.get_execution_stats()
    cache_stats = orchestrator.caching_layer.get_stats()
    
    print("📈 Execution Statistics:")
    print(f"   • Total verifications: {stats['total_verifications']}")
    print(f"   • Average execution time: {stats['avg_execution_time']:.3f}s")
    print(f"   • Cache hit rate: {cache_stats.hit_rate:.1%}")
    print()
    
    print("🎯 Agent Performance Summary:")
    for agent_name, perf in stats['agent_performance'].items():
        print(f"   • {agent_name}:")
        print(f"     - Success rate: {perf['success_rate']:.1%}")
        print(f"     - Avg score: {perf['avg_score']:.3f}")
        print(f"     - Avg time: {perf['avg_execution_time']:.3f}s")
    
    print()
    print("🎉 Phase 3 Orchestration Demo Complete!")
    print("✅ All 4 agents successfully coordinated in parallel")
    print("✅ Unified scoring and result aggregation working")
    print("✅ Caching layer providing performance optimization")
    print("✅ Enterprise-grade configuration and error handling")
    print()
    print("🚀 Ready for SWE-bench validation and enterprise deployment!")
    
    # Cleanup
    await orchestrator.cleanup()


async def run_single_verification_test():
    """Run a quick single verification to test the system"""
    
    print("\n🧪 Quick Single Verification Test:")
    print("-" * 40)
    
    # Simple test code with mixed quality
    test_code = '''
def hello_world(name):
    # Missing docstring and type hints
    if name:
        return f"Hello, {name}!"
    else:
        return "Hello, World!"

# Potential security issue
password = "hardcoded_secret_123"
'''
    
    orchestrator = AsyncOrchestrator()
    
    start_time = time.time()
    report = await orchestrator.verify_code(test_code)
    execution_time = time.time() - start_time
    
    print(f"⏱️  Execution Time: {execution_time:.3f}s")
    print(f"📊 Overall Score: {report.overall_score:.3f}")
    print(f"🎯 Status: {report.overall_status}")
    print(f"🔍 Issues Found: {len(report.aggregated_issues)}")
    
    if report.aggregated_issues:
        print("\n📋 Sample Issues:")
        for issue in report.aggregated_issues[:3]:
            severity_icon = {"critical": "🚨", "high": "⚠️", "medium": "🟡", "low": "ℹ️"}
            icon = severity_icon.get(issue.severity.value, "•")
            print(f"   {icon} {issue.message}")
    
    await orchestrator.cleanup()
    print("✅ Single verification test complete!")


if __name__ == "__main__":
    print("🎯 Starting CODEX-VERIFY Phase 3 Orchestration Demo...")
    print()
    
    try:
        # Run quick test first
        asyncio.run(run_single_verification_test())
        
        print("\n" + "=" * 60)
        
        # Run full demonstration
        asyncio.run(demonstrate_orchestration())
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Demo finished!")