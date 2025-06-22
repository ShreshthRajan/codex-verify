# src/orchestration/async_orchestrator.py
"""
Async Multi-Agent Orchestrator - Main coordination engine for all verification agents.
Coordinates parallel execution of all 4 agents with result aggregation and caching.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, asdict
from functools import lru_cache
import yaml
import os
import hashlib
import json
from pathlib import Path

# Fixed imports - use absolute imports
from agents.base_agent import BaseAgent, AgentResult, VerificationIssue, Severity
from agents.correctness_critic import CorrectnessAgent
from agents.security_auditor import SecurityAuditor
from agents.performance_profiler import PerformanceProfiler
from agents.style_maintainability import StyleMaintainabilityJudge
from orchestration.result_aggregator import ResultAggregator
from orchestration.caching_layer import CachingLayer


@dataclass
class VerificationConfig:
    """Configuration for verification process"""
    enabled_agents: Set[str]
    agent_configs: Dict[str, Dict[str, Any]]
    thresholds: Dict[str, float]
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour default
    max_execution_time: float = 60.0  # 60 seconds max
    parallel_execution: bool = True
    output_format: str = "json"  # json, yaml, text
    include_suggestions: bool = True
    include_metadata: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'VerificationConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls(
            enabled_agents=set(config_data.get('enabled_agents', ['all'])),
            agent_configs=config_data.get('agent_configs', {}),
            thresholds=config_data.get('thresholds', {}),
            enable_caching=config_data.get('enable_caching', True),
            cache_ttl=config_data.get('cache_ttl', 3600),
            max_execution_time=config_data.get('max_execution_time', 60.0),
            parallel_execution=config_data.get('parallel_execution', True),
            output_format=config_data.get('output_format', 'json'),
            include_suggestions=config_data.get('include_suggestions', True),
            include_metadata=config_data.get('include_metadata', True)
        )
    
    @classmethod
    def default(cls) -> 'VerificationConfig':
        """Create default configuration"""
        return cls(
            enabled_agents={'correctness', 'security', 'performance', 'style'},
            agent_configs={},
            thresholds={
                'correctness_min_score': 0.85,
                'security_min_score': 0.90,
                'performance_min_score': 0.80,
                'style_min_score': 0.85,
                'overall_min_score': 0.85
            }
        )


@dataclass
class VerificationReport:
    """Comprehensive verification report from all agents"""
    overall_score: float
    overall_status: str  # "PASS", "FAIL", "WARNING"
    execution_time: float
    agent_results: Dict[str, AgentResult]
    aggregated_issues: List[VerificationIssue]
    metadata: Dict[str, Any]
    summary: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            'overall_score': self.overall_score,
            'overall_status': self.overall_status,
            'execution_time': self.execution_time,
            'agent_results': {name: asdict(result) for name, result in self.agent_results.items()},
            'aggregated_issues': [asdict(issue) for issue in self.aggregated_issues],
            'metadata': self.metadata,
            'summary': self.summary,
            'recommendations': self.recommendations
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_yaml(self) -> str:
        """Convert report to YAML string"""
        return yaml.dump(self.to_dict(), default_flow_style=False)


class AsyncOrchestrator:
    """
    Main orchestration engine that coordinates all verification agents.
    
    Features:
    - Parallel agent execution with asyncio
    - Result aggregation and unified scoring
    - Intelligent caching for performance
    - Configurable thresholds and agent selection
    - Comprehensive error handling and recovery
    - Enterprise-grade reporting and metrics
    """
    
    def __init__(self, config: Optional[VerificationConfig] = None):
        self.config = config or VerificationConfig.default()
        self.result_aggregator = ResultAggregator(self.config)
        self.caching_layer = CachingLayer(enabled=self.config.enable_caching, ttl=self.config.cache_ttl)
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Execution statistics
        self.execution_stats = {
            'total_verifications': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_execution_time': 0.0,
            'agent_performance': {}
        }
    
    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """Initialize all verification agents with their configurations"""
        agents = {}
        
        # Map of agent names to classes
        agent_classes = {
            'correctness': CorrectnessAgent,
            'security': SecurityAuditor,
            'performance': PerformanceProfiler,
            'style': StyleMaintainabilityJudge
        }
        
        # Initialize enabled agents
        for agent_name in self.config.enabled_agents:
            if agent_name == 'all':
                # Enable all agents
                for name, agent_class in agent_classes.items():
                    agent_config = self.config.agent_configs.get(name, {})
                    agents[name] = agent_class(config=agent_config)
            elif agent_name in agent_classes:
                agent_config = self.config.agent_configs.get(agent_name, {})
                agents[agent_name] = agent_classes[agent_name](config=agent_config)
        
        return agents
    
    async def verify_code(self, code: str, context: Optional[Dict[str, Any]] = None) -> VerificationReport:
        """
        Main verification method - coordinates all agents and produces unified report.
        
        Args:
            code: Source code to verify
            context: Additional context (file_path, project_info, etc.)
            
        Returns:
            Comprehensive verification report
        """
        start_time = time.time()
        context = context or {}
        
        # Generate cache key
        cache_key = self._generate_cache_key(code, context)
        
        # Check cache first
        if self.config.enable_caching:
            cached_result = await self.caching_layer.get(cache_key)
            if cached_result:
                self.execution_stats['cache_hits'] += 1
                return cached_result
            self.execution_stats['cache_misses'] += 1
        
        try:
            # Execute verification
            if self.config.parallel_execution:
                agent_results = await self._execute_agents_parallel(code, context)
            else:
                agent_results = await self._execute_agents_sequential(code, context)
            
            # Aggregate results
            report = await self.result_aggregator.aggregate_results(
                agent_results, code, context, start_time
            )
            
            # Cache result
            if self.config.enable_caching:
                await self.caching_layer.set(cache_key, report)
            
            # Update statistics
            self._update_execution_stats(report, agent_results)
            
            return report
            
        except asyncio.TimeoutError:
            # Handle timeout
            return self._create_timeout_report(start_time)
        except Exception as e:
            # Handle unexpected errors
            return self._create_error_report(str(e), start_time)
    
    async def _execute_agents_parallel(self, code: str, context: Dict[str, Any]) -> Dict[str, AgentResult]:
        """Execute all agents in parallel using asyncio"""
        tasks = {}
        
        # Create tasks for each enabled agent
        for agent_name, agent in self.agents.items():
            task = asyncio.create_task(
                asyncio.wait_for(
                    agent.analyze(code, context),
                    timeout=self.config.max_execution_time / len(self.agents)
                )
            )
            tasks[agent_name] = task
        
        # Wait for all tasks to complete
        results = {}
        completed_tasks = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        for (agent_name, task), result in zip(tasks.items(), completed_tasks):
            if isinstance(result, Exception):
                # Handle individual agent failures
                results[agent_name] = AgentResult(
                    agent_name=agent_name,
                    execution_time=0.0,
                    overall_score=0.0,
                    issues=[],
                    metadata={'error': str(result)},
                    success=False,
                    error_message=str(result)
                )
            else:
                results[agent_name] = result
        
        return results
    
    async def _execute_agents_sequential(self, code: str, context: Dict[str, Any]) -> Dict[str, AgentResult]:
        """Execute agents sequentially (fallback mode)"""
        results = {}
        
        for agent_name, agent in self.agents.items():
            try:
                result = await asyncio.wait_for(
                    agent.analyze(code, context),
                    timeout=self.config.max_execution_time / len(self.agents)
                )
                results[agent_name] = result
            except Exception as e:
                results[agent_name] = AgentResult(
                    agent_name=agent_name,
                    execution_time=0.0,
                    overall_score=0.0,
                    issues=[],
                    metadata={'error': str(e)},
                    success=False,
                    error_message=str(e)
                )
        
        return results
    
    def _generate_cache_key(self, code: str, context: Dict[str, Any]) -> str:
        """Generate cache key for code and context"""
        # Create hash of code content
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        
        # Create hash of relevant context
        context_str = json.dumps(context, sort_keys=True)
        context_hash = hashlib.sha256(context_str.encode()).hexdigest()
        
        # Include agent configuration in key
        agent_config_str = json.dumps(self.config.agent_configs, sort_keys=True)
        config_hash = hashlib.sha256(agent_config_str.encode()).hexdigest()
        
        return f"{code_hash}_{context_hash}_{config_hash}"
    
    def _create_timeout_report(self, start_time: float) -> VerificationReport:
        """Create report for timeout scenarios"""
        return VerificationReport(
            overall_score=0.0,
            overall_status="TIMEOUT",
            execution_time=time.time() - start_time,
            agent_results={},
            aggregated_issues=[
                VerificationIssue(
                    type="timeout",
                    severity=Severity.CRITICAL,
                    message=f"Verification timed out after {self.config.max_execution_time}s",
                    suggestion="Reduce code complexity or increase timeout limit"
                )
            ],
            metadata={'timeout': True, 'max_execution_time': self.config.max_execution_time},
            summary={'status': 'timeout', 'reason': 'Execution exceeded time limit'},
            recommendations=['Optimize code for faster analysis', 'Increase timeout configuration']
        )
    
    def _create_error_report(self, error_message: str, start_time: float) -> VerificationReport:
        """Create report for error scenarios"""
        return VerificationReport(
            overall_score=0.0,
            overall_status="ERROR",
            execution_time=time.time() - start_time,
            agent_results={},
            aggregated_issues=[
                VerificationIssue(
                    type="system_error",
                    severity=Severity.CRITICAL,
                    message=f"Verification failed: {error_message}",
                    suggestion="Check code syntax and system configuration"
                )
            ],
            metadata={'error': True, 'error_message': error_message},
            summary={'status': 'error', 'reason': error_message},
            recommendations=['Check code syntax', 'Review system configuration', 'Contact support if issue persists']
        )
    
    def _update_execution_stats(self, report: VerificationReport, agent_results: Dict[str, AgentResult]):
        """Update execution statistics"""
        self.execution_stats['total_verifications'] += 1
        
        # Update average execution time
        total_time = self.execution_stats['avg_execution_time'] * (self.execution_stats['total_verifications'] - 1)
        total_time += report.execution_time
        self.execution_stats['avg_execution_time'] = total_time / self.execution_stats['total_verifications']
        
        # Update agent performance stats
        for agent_name, result in agent_results.items():
            if agent_name not in self.execution_stats['agent_performance']:
                self.execution_stats['agent_performance'][agent_name] = {
                    'total_executions': 0,
                    'avg_execution_time': 0.0,
                    'success_rate': 0.0,
                    'avg_score': 0.0
                }
            
            stats = self.execution_stats['agent_performance'][agent_name]
            stats['total_executions'] += 1
            
            # Update average execution time
            total_time = stats['avg_execution_time'] * (stats['total_executions'] - 1)
            total_time += result.execution_time
            stats['avg_execution_time'] = total_time / stats['total_executions']
            
            # Update success rate
            total_successes = stats['success_rate'] * (stats['total_executions'] - 1)
            total_successes += (1 if result.success else 0)
            stats['success_rate'] = total_successes / stats['total_executions']
            
            # Update average score
            total_score = stats['avg_score'] * (stats['total_executions'] - 1)
            total_score += result.overall_score
            stats['avg_score'] = total_score / stats['total_executions']
    
    async def batch_verify(self, code_samples: List[Dict[str, Any]]) -> List[VerificationReport]:
        """
        Verify multiple code samples in batch.
        
        Args:
            code_samples: List of dicts with 'code' and optional 'context' keys
            
        Returns:
            List of verification reports
        """
        if self.config.parallel_execution:
            tasks = [
                self.verify_code(sample['code'], sample.get('context', {}))
                for sample in code_samples
            ]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for sample in code_samples:
                result = await self.verify_code(sample['code'], sample.get('context', {}))
                results.append(result)
            return results
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get current execution statistics"""
        return self.execution_stats.copy()
    
    def reset_stats(self):
        """Reset execution statistics"""
        self.execution_stats = {
            'total_verifications': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_execution_time': 0.0,
            'agent_performance': {}
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all agents"""
        health_status = {
            'overall_status': 'healthy',
            'agents': {},
            'cache': await self.caching_layer.health_check(),
            'timestamp': time.time()
        }
        
        # Test each agent with simple code
        test_code = "def hello(): return 'world'"
        
        for agent_name, agent in self.agents.items():
            try:
                start_time = time.time()
                result = await asyncio.wait_for(agent.analyze(test_code), timeout=5.0)
                health_status['agents'][agent_name] = {
                    'status': 'healthy',
                    'response_time': time.time() - start_time,
                    'last_score': result.overall_score
                }
            except Exception as e:
                health_status['agents'][agent_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_status['overall_status'] = 'degraded'
        
        return health_status
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.caching_layer.cleanup()


# Factory function for easy instantiation
def create_orchestrator(config_path: Optional[str] = None, **kwargs) -> AsyncOrchestrator:
    """
    Factory function to create orchestrator with optional configuration.
    
    Args:
        config_path: Path to YAML configuration file
        **kwargs: Additional configuration overrides
        
    Returns:
        Configured AsyncOrchestrator instance
    """
    if config_path and os.path.exists(config_path):
        config = VerificationConfig.from_yaml(config_path)
    else:
        config = VerificationConfig.default()
    
    # Apply any kwargs overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return AsyncOrchestrator(config)

def to_json(self, indent=2):
    """Convert result to JSON string"""
    import json
    return json.dumps({
        'overall_score': self.overall_score,
        'overall_status': self.overall_status,
        'execution_time': self.execution_time,
        'agent_results': {
            name: {
                'score': result.overall_score,
                'issues': len(result.issues),
                'success': result.success,
                'execution_time': result.execution_time
            }
            for name, result in self.agent_results.items()
        },
        'total_issues': len(self.aggregated_issues),
        'critical_issues': len([i for i in self.aggregated_issues if i.severity.value == 'critical']),
        'high_issues': len([i for i in self.aggregated_issues if i.severity.value == 'high']),
        'timestamp': time.time()
    }, indent=indent, default=str)

def to_yaml(self):
    """Convert result to YAML string"""
    import yaml
    data = {
        'verification_report': {
            'overall_score': f"{self.overall_score:.1%}",
            'status': self.overall_status,
            'execution_time': f"{self.execution_time:.3f}s",
            'agents': {
                name: {
                    'score': f"{result.overall_score:.1%}",
                    'issues_found': len(result.issues),
                    'execution_time': f"{result.execution_time:.3f}s"
                }
                for name, result in self.agent_results.items()
            },
            'summary': {
                'total_issues': len(self.aggregated_issues),
                'critical_issues': len([i for i in self.aggregated_issues if i.severity.value == 'critical']),
                'high_issues': len([i for i in self.aggregated_issues if i.severity.value == 'high'])
            }
        }
    }
    return yaml.dump(data, default_flow_style=False)

def to_dict(self):
    """Convert result to dictionary"""
    return {
        'overall_score': self.overall_score,
        'overall_status': self.overall_status,
        'execution_time': self.execution_time,
        'agent_results': {
            name: {
                'score': result.overall_score,
                'issues': [
                    {
                        'type': issue.type,
                        'severity': issue.severity.value,
                        'message': issue.message,
                        'line_number': issue.line_number,
                        'suggestion': issue.suggestion
                    }
                    for issue in result.issues
                ],
                'metadata': result.metadata,
                'success': result.success,
                'execution_time': result.execution_time
            }
            for name, result in self.agent_results.items()
        },
        'aggregated_issues': [
            {
                'type': issue.type,
                'severity': issue.severity.value,
                'message': issue.message,
                'line_number': issue.line_number,
                'suggestion': issue.suggestion
            }
            for issue in self.aggregated_issues
        ]
    }