"""
Base agent class for all verification agents.
Provides common interface and utilities for multi-agent verification framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
import time
import asyncio


class Severity(Enum):
    """Severity levels for verification issues"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class VerificationIssue:
    """Individual verification issue found by an agent"""
    type: str
    severity: Severity
    message: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    file_path: Optional[str] = None
    suggestion: Optional[str] = None
    confidence: float = 1.0


@dataclass
class AgentResult:
    """Result from a verification agent"""
    agent_name: str
    execution_time: float
    overall_score: float  # 0.0 to 1.0, higher is better
    issues: List[VerificationIssue]
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None


class BaseAgent(ABC):
    """Abstract base class for all verification agents"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
    
    async def analyze(self, code: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Main analysis method that all agents must implement.
        Handles timing and error management.
        """
        if not self.enabled:
            return AgentResult(
                agent_name=self.name,
                execution_time=0.0,
                overall_score=1.0,
                issues=[],
                metadata={"skipped": True}
            )
        
        start_time = time.time()
        context = context or {}
        
        try:
            result = await self._analyze_implementation(code, context)
            execution_time = time.time() - start_time
            
            return AgentResult(
                agent_name=self.name,
                execution_time=execution_time,
                overall_score=result.overall_score,
                issues=result.issues,
                metadata=result.metadata,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AgentResult(
                agent_name=self.name,
                execution_time=execution_time,
                overall_score=0.0,
                issues=[],
                metadata={"error_details": str(e)},
                success=False,
                error_message=str(e)
            )
    
    @abstractmethod
    async def _analyze_implementation(self, code: str, context: Dict[str, Any]) -> AgentResult:
        """Implementation-specific analysis logic"""
        pass
    
    def _calculate_score(self, issues: List[VerificationIssue]) -> float:
        """Calculate overall score based on issues found"""
        if not issues:
            return 1.0
        
        # Weight issues by severity
        severity_weights = {
            Severity.LOW: 0.1,
            Severity.MEDIUM: 0.3,
            Severity.HIGH: 0.7,
            Severity.CRITICAL: 1.0
        }
        
        total_weight = sum(severity_weights[issue.severity] for issue in issues)
        max_possible_weight = len(issues) * severity_weights[Severity.CRITICAL]
        
        # Score decreases as more severe issues are found
        return max(0.0, 1.0 - (total_weight / max(max_possible_weight, 1.0)))