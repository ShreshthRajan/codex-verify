# src/orchestration/result_aggregator.py
"""
Result Aggregator - Combines outputs from all verification agents into unified reports.
Handles result aggregation, scoring, deduplication, and comprehensive metadata generation.
"""

import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import statistics

from ..agents.base_agent import AgentResult, VerificationIssue, Severity


@dataclass
class IssueCluster:
    """Cluster of similar issues from different agents"""
    primary_issue: VerificationIssue
    related_issues: List[VerificationIssue]
    agent_sources: Set[str]
    confidence: float
    severity: Severity


class ResultAggregator:
    """
    Aggregates and processes results from multiple verification agents.
    
    Features:
    - Issue deduplication and clustering
    - Unified scoring algorithm
    - Comprehensive metadata generation
    - Priority-based recommendations
    - Report formatting and export
    """
    
    def __init__(self, config):
        self.config = config
        
        # Scoring weights for different agents
        self.agent_weights = {
            'correctness': 0.30,    # Correctness is critical
            'security': 0.35,       # Security is most important  
            'performance': 0.20,    # Performance matters for production
            'style': 0.15          # Style is important but less critical
        }
        
        # Issue type priorities for recommendation generation
        self.issue_priorities = {
            'code_execution': 1,    # Critical security
            'sql_injection': 1,
            'secret': 1,
            'syntax_error': 2,      # Correctness issues
            'logic_error': 2,
            'performance_critical': 3,  # Performance issues
            'complexity': 4,
            'style': 5             # Style issues
        }
    
    async def aggregate_results(self, agent_results: Dict[str, AgentResult], 
                              code: str, context: Dict[str, Any], 
                              start_time: float) -> 'VerificationReport':
        """
        Main aggregation method that combines all agent results.
        
        Args:
            agent_results: Results from all executed agents
            code: Original code that was analyzed
            context: Analysis context
            start_time: Start time of verification process
            
        Returns:
            Unified verification report
        """
        from .async_orchestrator import VerificationReport  # Import here to avoid circular imports
        
        execution_time = time.time() - start_time
        
        # Extract all issues from agents
        all_issues = []
        for agent_name, result in agent_results.items():
            for issue in result.issues:
                # Add agent source to issue metadata
                issue_dict = issue.__dict__.copy()
                issue_dict['agent_source'] = agent_name
                all_issues.append(VerificationIssue(**issue_dict))
        
        # Deduplicate and cluster similar issues
        clustered_issues = self._cluster_similar_issues(all_issues)
        
        # Calculate unified score
        overall_score = self._calculate_unified_score(agent_results)
        
        # Determine overall status
        overall_status = self._determine_status(overall_score, clustered_issues)
        
        # Generate comprehensive metadata
        metadata = self._generate_metadata(agent_results, clustered_issues, code, context)
        
        # Create summary
        summary = self._create_summary(agent_results, clustered_issues, overall_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(clustered_issues, agent_results)
        
        return VerificationReport(
            overall_score=overall_score,
            overall_status=overall_status,
            execution_time=execution_time,
            agent_results=agent_results,
            aggregated_issues=clustered_issues,
            metadata=metadata,
            summary=summary,
            recommendations=recommendations
        )
    
    def _cluster_similar_issues(self, issues: List[VerificationIssue]) -> List[VerificationIssue]:
        """
        Cluster similar issues to reduce duplication and noise.
        
        Args:
            issues: All issues from all agents
            
        Returns:
            Deduplicated and clustered issues
        """
        if not issues:
            return []
        
        # Group issues by type and line number
        grouped_issues = defaultdict(list)
        
        for issue in issues:
            # Create grouping key based on type, line, and similar message content
            key_parts = [issue.type]
            
            if issue.line_number:
                key_parts.append(f"line_{issue.line_number}")
            
            # Group by similar message content (first few words)
            message_key = "_".join(issue.message.lower().split()[:3])
            key_parts.append(message_key)
            
            group_key = "_".join(key_parts)
            grouped_issues[group_key].append(issue)
        
        # Process each group
        clustered_issues = []
        
        for group_key, group_issues in grouped_issues.items():
            if len(group_issues) == 1:
                # Single issue, keep as-is
                clustered_issues.append(group_issues[0])
            else:
                # Multiple similar issues, create merged issue
                merged_issue = self._merge_similar_issues(group_issues)
                clustered_issues.append(merged_issue)
        
        # Sort by severity and priority
        clustered_issues.sort(key=lambda x: (
            self._severity_to_numeric(x.severity),
            self.issue_priorities.get(x.type, 10),
            x.line_number or 0
        ), reverse=True)
        
        return clustered_issues
    
    def _merge_similar_issues(self, issues: List[VerificationIssue]) -> VerificationIssue:
        """
        Merge multiple similar issues into a single representative issue.
        
        Args:
            issues: List of similar issues to merge
            
        Returns:
            Merged issue representing the group
        """
        # Use the issue with highest severity as primary
        primary_issue = max(issues, key=lambda x: self._severity_to_numeric(x.severity))
        
        # Collect agent sources
        agent_sources = set()
        for issue in issues:
            if hasattr(issue, 'agent_source'):
                agent_sources.add(issue.agent_source)
        
        # Calculate average confidence
        confidences = [issue.confidence for issue in issues if hasattr(issue, 'confidence')]
        avg_confidence = statistics.mean(confidences) if confidences else 1.0
        
        # Create merged message if multiple agents found the same issue
        if len(agent_sources) > 1:
            message_suffix = f" (detected by {len(agent_sources)} agents: {', '.join(sorted(agent_sources))})"
            merged_message = primary_issue.message + message_suffix
        else:
            merged_message = primary_issue.message
        
        # Create merged issue
        merged_issue = VerificationIssue(
            type=primary_issue.type,
            severity=primary_issue.severity,
            message=merged_message,
            line_number=primary_issue.line_number,
            column_number=primary_issue.column_number,
            file_path=primary_issue.file_path,
            suggestion=primary_issue.suggestion,
            confidence=avg_confidence
        )
        
        # Add metadata about merging
        merged_issue.merged_count = len(issues)
        merged_issue.agent_sources = agent_sources
        
        return merged_issue
    
    def _severity_to_numeric(self, severity: Severity) -> int:
        """Convert severity to numeric value for sorting"""
        severity_map = {
            Severity.LOW: 1,
            Severity.MEDIUM: 2,
            Severity.HIGH: 3,
            Severity.CRITICAL: 4
        }
        return severity_map.get(severity, 0)
    
    def _calculate_unified_score(self, agent_results: Dict[str, AgentResult]) -> float:
        """
        Calculate unified verification score from all agent results.
        
        Uses weighted average based on agent importance and success status.
        """
        if not agent_results:
            return 0.0
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for agent_name, result in agent_results.items():
            # Get agent weight
            weight = self.agent_weights.get(agent_name, 0.25)
            
            # Adjust weight based on success
            if not result.success:
                # Failed agents get zero contribution but still count toward total
                agent_score = 0.0
            else:
                agent_score = result.overall_score
            
            weighted_score += agent_score * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0.0
        
        # Apply penalty for failed agents
        failed_agents = sum(1 for result in agent_results.values() if not result.success)
        if failed_agents > 0:
            failure_penalty = 0.1 * failed_agents  # 10% penalty per failed agent
            final_score = max(0.0, final_score - failure_penalty)
        
        return round(final_score, 3)
    
    def _determine_status(self, overall_score: float, issues: List[VerificationIssue]) -> str:
        """
        Determine overall verification status based on score and issues.
        
        Args:
            overall_score: Calculated overall score
            issues: All aggregated issues
            
        Returns:
            Status string: "PASS", "FAIL", "WARNING"
        """
        # Check for critical issues first
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        if critical_issues:
            return "FAIL"
        
        # Check against configured thresholds
        min_score = self.config.thresholds.get('overall_min_score', 0.85)
        
        if overall_score >= min_score:
            # High score but check for high severity issues
            high_issues = [i for i in issues if i.severity == Severity.HIGH]
            if len(high_issues) > 3:  # More than 3 high-severity issues
                return "WARNING"
            return "PASS"
        elif overall_score >= min_score * 0.7:  # Within 30% of threshold
            return "WARNING"
        else:
            return "FAIL"
    
    def _generate_metadata(self, agent_results: Dict[str, AgentResult], 
                          issues: List[VerificationIssue], 
                          code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive metadata for the verification report"""
        metadata = {
            'analysis_timestamp': time.time(),
            'code_stats': self._analyze_code_stats(code),
            'agent_summary': {},
            'issue_summary': self._summarize_issues(issues),
            'quality_metrics': {},
            'context': context
        }
        
        # Agent-specific metadata
        for agent_name, result in agent_results.items():
            metadata['agent_summary'][agent_name] = {
                'success': result.success,
                'execution_time': result.execution_time,
                'score': result.overall_score,
                'issue_count': len(result.issues),
                'metadata': result.metadata
            }
        
        # Quality metrics aggregation
        metadata['quality_metrics'] = self._calculate_quality_metrics(agent_results)
        
        # Performance analysis
        metadata['performance'] = {
            'total_agents': len(agent_results),
            'successful_agents': sum(1 for r in agent_results.values() if r.success),
            'total_execution_time': sum(r.execution_time for r in agent_results.values()),
            'avg_agent_time': statistics.mean([r.execution_time for r in agent_results.values()]) if agent_results else 0
        }
        
        return metadata
    
    def _analyze_code_stats(self, code: str) -> Dict[str, Any]:
        """Analyze basic code statistics"""
        lines = code.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        comment_lines = [line for line in lines if line.strip().startswith('#')]
        
        return {
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'comment_lines': len(comment_lines),
            'blank_lines': len(lines) - len(non_empty_lines),
            'comment_ratio': len(comment_lines) / len(non_empty_lines) if non_empty_lines else 0,
            'characters': len(code),
            'avg_line_length': statistics.mean([len(line) for line in lines]) if lines else 0
        }
    
    def _summarize_issues(self, issues: List[VerificationIssue]) -> Dict[str, Any]:
        """Summarize issues by type and severity"""
        if not issues:
            return {'total': 0, 'by_severity': {}, 'by_type': {}}
        
        severity_counts = Counter(issue.severity.value for issue in issues)
        type_counts = Counter(issue.type for issue in issues)
        
        return {
            'total': len(issues),
            'by_severity': dict(severity_counts),
            'by_type': dict(type_counts),
            'critical_count': sum(1 for i in issues if i.severity == Severity.CRITICAL),
            'high_count': sum(1 for i in issues if i.severity == Severity.HIGH),
            'unique_types': len(type_counts)
        }
    
    def _calculate_quality_metrics(self, agent_results: Dict[str, AgentResult]) -> Dict[str, Any]:
        """Calculate aggregated quality metrics from all agents"""
        metrics = {
            'overall_health': 0.0,
            'complexity_score': 0.0,
            'security_score': 0.0,
            'maintainability_score': 0.0,
            'documentation_score': 0.0
        }
        
        # Extract specific metrics from agent results
        for agent_name, result in agent_results.items():
            if not result.success:
                continue
                
            agent_metadata = result.metadata
            
            # Security metrics
            if agent_name == 'security' and 'security_metrics' in agent_metadata:
                sec_metrics = agent_metadata['security_metrics']
                if isinstance(sec_metrics, dict):
                    # Convert risk score to 0-1 scale (lower risk = higher score)
                    risk_score = sec_metrics.get('total_risk_score', 0)
                    metrics['security_score'] = max(0.0, 1.0 - min(risk_score / 50.0, 1.0))
            
            # Performance/complexity metrics
            elif agent_name == 'performance' and 'performance_metrics' in agent_metadata:
                perf_metrics = agent_metadata['performance_metrics']
                if isinstance(perf_metrics, dict):
                    # Use complexity metrics
                    complexity = perf_metrics.get('avg_complexity', 0)
                    metrics['complexity_score'] = max(0.0, 1.0 - min(complexity / 20.0, 1.0))
            
            # Style/maintainability metrics
            elif agent_name == 'style':
                # Extract maintainability from style agent
                if 'maintainability_metrics' in agent_metadata:
                    maint_metrics = agent_metadata['maintainability_metrics']
                    if isinstance(maint_metrics, dict):
                        mi = maint_metrics.get('maintainability_index', 0)
                        metrics['maintainability_score'] = mi / 100.0  # MI is 0-100 scale
                
                # Extract documentation score
                if 'documentation_metrics' in agent_metadata:
                    doc_metrics = agent_metadata['documentation_metrics']
                    if isinstance(doc_metrics, dict):
                        metrics['documentation_score'] = doc_metrics.get('docstring_coverage', 0)
        
        # Calculate overall health as weighted average
        weights = {'security_score': 0.3, 'complexity_score': 0.25, 
                  'maintainability_score': 0.25, 'documentation_score': 0.2}
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric, weight in weights.items():
            if metrics[metric] > 0:  # Only include metrics we have data for
                weighted_sum += metrics[metric] * weight
                total_weight += weight
        
        metrics['overall_health'] = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return metrics
    
    def _create_summary(self, agent_results: Dict[str, AgentResult], 
                       issues: List[VerificationIssue], 
                       overall_score: float) -> Dict[str, Any]:
        """Create comprehensive summary of verification results"""
        
        # Agent performance summary
        agent_summary = {}
        for agent_name, result in agent_results.items():
            status = "✓ PASS" if result.success and result.overall_score >= 0.8 else "✗ FAIL"
            agent_summary[agent_name] = {
                'status': status,
                'score': result.overall_score,
                'issues_found': len(result.issues),
                'execution_time': f"{result.execution_time:.3f}s"
            }
        
        # Issue breakdown
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        high_issues = [i for i in issues if i.severity == Severity.HIGH]
        medium_issues = [i for i in issues if i.severity == Severity.MEDIUM]
        low_issues = [i for i in issues if i.severity == Severity.LOW]
        
        # Key findings
        key_findings = []
        if critical_issues:
            key_findings.append(f"{len(critical_issues)} critical security/correctness issues")
        if high_issues:
            key_findings.append(f"{len(high_issues)} high-priority issues")
        if len(issues) == 0:
            key_findings.append("No issues detected - excellent code quality")
        
        # Pass/fail reasons
        pass_fail_reasons = []
        if overall_score >= 0.9:
            pass_fail_reasons.append("High overall quality score")
        elif critical_issues:
            pass_fail_reasons.append("Critical issues must be resolved")
        elif overall_score < 0.7:
            pass_fail_reasons.append("Overall score below acceptable threshold")
        
        return {
            'overall_score': overall_score,
            'grade': self._score_to_grade(overall_score),
            'agent_performance': agent_summary,
            'issue_breakdown': {
                'critical': len(critical_issues),
                'high': len(high_issues),
                'medium': len(medium_issues),
                'low': len(low_issues),
                'total': len(issues)
            },
            'key_findings': key_findings,
            'pass_fail_reasons': pass_fail_reasons,
            'top_issue_types': self._get_top_issue_types(issues)
        }
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 0.95:
            return "A+"
        elif score >= 0.90:
            return "A"
        elif score >= 0.85:
            return "A-"
        elif score >= 0.80:
            return "B+"
        elif score >= 0.75:
            return "B"
        elif score >= 0.70:
            return "B-"
        elif score >= 0.65:
            return "C+"
        elif score >= 0.60:
            return "C"
        elif score >= 0.50:
            return "D"
        else:
            return "F"
    
    def _get_top_issue_types(self, issues: List[VerificationIssue], top_n: int = 5) -> List[Dict[str, Any]]:
        """Get top N most common issue types"""
        if not issues:
            return []
        
        type_counts = Counter(issue.type for issue in issues)
        top_types = []
        
        for issue_type, count in type_counts.most_common(top_n):
            # Get worst severity for this type
            type_issues = [i for i in issues if i.type == issue_type]
            worst_severity = max(type_issues, key=lambda x: self._severity_to_numeric(x.severity)).severity
            
            top_types.append({
                'type': issue_type,
                'count': count,
                'worst_severity': worst_severity.value
            })
        
        return top_types
    
    def _generate_recommendations(self, issues: List[VerificationIssue], 
                                agent_results: Dict[str, AgentResult]) -> List[str]:
        """Generate prioritized recommendations based on analysis results"""
        recommendations = []
        
        # Critical issues first
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        if critical_issues:
            recommendations.append("🚨 URGENT: Resolve critical security/correctness issues immediately")
            for issue in critical_issues[:3]:  # Top 3 critical issues
                if issue.suggestion:
                    recommendations.append(f"   • {issue.suggestion}")
        
        # Security recommendations
        security_issues = [i for i in issues if i.type in ['vulnerability', 'secret', 'code_execution']]
        if security_issues:
            recommendations.append("🔒 Security: Address security vulnerabilities")
            unique_suggestions = set()
            for issue in security_issues:
                if issue.suggestion and issue.suggestion not in unique_suggestions:
                    recommendations.append(f"   • {issue.suggestion}")
                    unique_suggestions.add(issue.suggestion)
                    if len(unique_suggestions) >= 2:  # Limit suggestions
                        break
        
        # Performance recommendations
        perf_agent = agent_results.get('performance')
        if perf_agent and perf_agent.success and perf_agent.overall_score < 0.7:
            recommendations.append("⚡ Performance: Optimize code efficiency")
            perf_issues = [i for i in issues if 'performance' in i.type or 'complexity' in i.type]
            for issue in perf_issues[:2]:
                if issue.suggestion:
                    recommendations.append(f"   • {issue.suggestion}")
        
        # Documentation recommendations
        style_agent = agent_results.get('style')
        if style_agent and style_agent.success:
            doc_metrics = style_agent.metadata.get('documentation_metrics', {})
            if isinstance(doc_metrics, dict) and doc_metrics.get('docstring_coverage', 1.0) < 0.8:
                recommendations.append("📚 Documentation: Improve code documentation")
                recommendations.append("   • Add docstrings to functions and classes")
                recommendations.append("   • Include more explanatory comments")
        
        # Code quality recommendations
        style_issues = [i for i in issues if i.type in ['style', 'maintainability_index', 'consistency']]
        if len(style_issues) > 5:
            recommendations.append("🎨 Code Quality: Improve code style and maintainability")
            recommendations.append("   • Use consistent formatting (consider black or autopep8)")
            recommendations.append("   • Follow PEP 8 style guidelines")
        
        # General improvement recommendations
        if len(issues) > 10:
            recommendations.append("🔄 General: Consider refactoring to reduce complexity")
        
        # Success recommendations
        if len(issues) == 0:
            recommendations.extend([
                "✅ Excellent! Your code meets all quality standards",
                "💡 Consider adding more tests for edge cases",
                "📈 Review performance in production environments"
            ])
        elif len(critical_issues) == 0 and len([i for i in issues if i.severity == Severity.HIGH]) == 0:
            recommendations.append("👍 Good code quality with only minor issues to address")
        
        return recommendations[:10]  # Limit to top 10 recommendations