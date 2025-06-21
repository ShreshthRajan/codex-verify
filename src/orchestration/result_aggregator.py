# src/orchestration/result_aggregator.py
"""
Result Aggregator - Enterprise-grade result processing with production deployment standards.
Implements compound vulnerability detection and scale-aware verification scoring.
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
    Enterprise-grade result aggregator with production deployment standards.
    
    Breakthrough Features:
    - Compound vulnerability detection with exponential penalties
    - Security veto power for production deployment
    - Scale-aware algorithmic scoring
    - Enterprise production readiness enforcement
    """
    
    def __init__(self, config):
        self.config = config
        
        # Enterprise-weighted agent importance
        self.agent_weights = {
            'security': 0.40,       # Security is paramount for production
            'correctness': 0.30,    # Correctness blocks deployment
            'performance': 0.20,    # Performance affects scale
            'style': 0.10          # Style affects maintainability
        }
        
        # Production deployment blocker priorities
        self.production_blockers = {
            'code_execution': 1,     # Immediate security threat
            'sql_injection': 1,      # Data breach risk
            'secret': 1,             # Credential exposure
            'unsafe_deserialization': 1,  # Code execution vector
            'complexity': 2,         # Maintenance/scale risk
            'algorithm_inefficiency': 2,  # Performance at scale
            'resource_leak': 3,      # Production stability
            'edge_case_missing': 3,  # Runtime failures
        }
        
        # Compound vulnerability multipliers
        self.compound_multipliers = {
            ('vulnerability', 'secret'): 2.5,           # Security + exposure
            ('sql_injection', 'hardcoded_secret'): 3.0, # DB + credentials
            ('code_execution', 'dangerous_import'): 2.0, # Execution + tools
            ('complexity', 'algorithm_inefficiency'): 1.8, # Scale nightmare
            ('memory_leak', 'performance_critical'): 1.5   # Resource issues
        }
    
    async def aggregate_results(self, agent_results: Dict[str, AgentResult], 
                              code: str, context: Dict[str, Any], 
                              start_time: float) -> 'VerificationReport':
        """
        Enterprise aggregation with production deployment standards.
        """
        from .async_orchestrator import VerificationReport
        
        execution_time = time.time() - start_time
        
        # Extract and enrich issues
        all_issues = self._extract_and_enrich_issues(agent_results)
        
        # Detect compound vulnerabilities 
        compound_issues = self._detect_compound_vulnerabilities(all_issues)
        all_issues.extend(compound_issues)
        
        # Cluster similar issues
        clustered_issues = self._cluster_similar_issues(all_issues)
        
        # Calculate enterprise score with production standards
        overall_score = self._calculate_enterprise_score(agent_results, clustered_issues)
        
        # Determine production deployment status
        overall_status = self._determine_production_status(overall_score, clustered_issues)
        
        # Generate metadata
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
    
    def _extract_and_enrich_issues(self, agent_results: Dict[str, AgentResult]) -> List[VerificationIssue]:
        """Extract issues and enrich with agent context"""
        all_issues = []
        
        for agent_name, result in agent_results.items():
            for issue in result.issues:
                # Enrich issue with agent source and context
                issue_dict = issue.__dict__.copy()
                issue_dict['agent_source'] = agent_name
                
                # Apply enterprise context-aware severity adjustment
                adjusted_severity = self._adjust_severity_for_context(issue, agent_name, result.metadata)
                issue_dict['severity'] = adjusted_severity
                
                enriched_issue = VerificationIssue(**issue_dict)
                all_issues.append(enriched_issue)
        
        return all_issues
    
    def _adjust_severity_for_context(self, issue: VerificationIssue, agent_name: str, 
                                   metadata: Dict[str, Any]) -> Severity:
        """Apply enterprise context-aware severity adjustment"""
        current_severity = issue.severity
        
        # Security context escalation
        if agent_name == 'security':
            # Crypto vulnerabilities in authentication context = CRITICAL
            if 'weak_hash' in issue.type and 'auth' in issue.message.lower():
                return Severity.CRITICAL
            
            # SQL injection with database operations = CRITICAL  
            if 'sql' in issue.type and any(word in issue.message.lower() 
                                         for word in ['user', 'auth', 'login', 'password']):
                return Severity.CRITICAL
            
            # Multiple secrets detected = escalate severity
            if 'secret' in issue.type:
                secret_metrics = metadata.get('security_metrics', {})
                if isinstance(secret_metrics, dict) and secret_metrics.get('secrets_found', 0) > 2:
                    return Severity.CRITICAL if current_severity == Severity.HIGH else Severity.HIGH
        
        # Performance context escalation
        elif agent_name == 'performance':
            # O(n²) in sorting/searching algorithms = CRITICAL for production
            if 'complexity' in issue.type and any(word in issue.message.lower() 
                                                for word in ['sort', 'search', 'find', 'query']):
                return Severity.CRITICAL
            
            # Memory leaks in long-running processes = CRITICAL
            if 'memory' in issue.type or 'resource' in issue.type:
                return Severity.HIGH if current_severity == Severity.MEDIUM else current_severity
        
        # Correctness context escalation  
        elif agent_name == 'correctness':
            # Missing exception handling in critical operations = HIGH
            if 'exception' in issue.message.lower() or 'error' in issue.message.lower():
                return Severity.HIGH if current_severity == Severity.MEDIUM else current_severity
        
        return current_severity
    
    def _detect_compound_vulnerabilities(self, issues: List[VerificationIssue]) -> List[VerificationIssue]:
        """Detect compound vulnerabilities that create exponential risk"""
        compound_issues = []
        issue_types = [issue.type for issue in issues]
        
        # Check for dangerous combinations
        for (type1, type2), multiplier in self.compound_multipliers.items():
            if type1 in issue_types and type2 in issue_types:
                compound_issues.append(VerificationIssue(
                    type="compound_vulnerability",
                    severity=Severity.CRITICAL,
                    message=f"Compound vulnerability: {type1} + {type2} (risk multiplier: {multiplier}x)",
                    suggestion=f"Address both {type1} and {type2} immediately - combined risk is {multiplier}x normal",
                    confidence=0.95
                ))
        
        # Special detection: Multiple security issues = production blocker
        security_issues = [i for i in issues if hasattr(i, 'agent_source') and i.agent_source == 'security']
        critical_security = [i for i in security_issues if i.severity == Severity.CRITICAL]
        high_security = [i for i in security_issues if i.severity == Severity.HIGH]
        
        if len(critical_security) >= 1 and len(high_security) >= 1:
            compound_issues.append(VerificationIssue(
                type="security_cascade",
                severity=Severity.CRITICAL,
                message=f"Security cascade: {len(critical_security)} critical + {len(high_security)} high security issues",
                suggestion="Multiple security vulnerabilities create unacceptable production risk",
                confidence=1.0
            ))
        
        return compound_issues
    
    def _cluster_similar_issues(self, issues: List[VerificationIssue]) -> List[VerificationIssue]:
        """Cluster similar issues while preserving severity escalation"""
        if not issues:
            return []
        
        # Group issues by type and context
        grouped_issues = defaultdict(list)
        
        for issue in issues:
            # Create intelligent grouping key
            key_parts = [issue.type]
            
            if issue.line_number:
                key_parts.append(f"line_{issue.line_number}")
            
            # Group by severity and agent source for enterprise analysis
            if hasattr(issue, 'agent_source'):
                key_parts.append(issue.agent_source)
            
            key_parts.append(issue.severity.value)
            
            group_key = "_".join(key_parts)
            grouped_issues[group_key].append(issue)
        
        # Process groups with enterprise logic
        clustered_issues = []
        
        for group_key, group_issues in grouped_issues.items():
            if len(group_issues) == 1:
                clustered_issues.append(group_issues[0])
            else:
                # Merge with severity escalation
                merged_issue = self._merge_with_escalation(group_issues)
                clustered_issues.append(merged_issue)
        
        # Sort by enterprise priority
        clustered_issues.sort(key=lambda x: (
            self._severity_to_numeric(x.severity),
            self.production_blockers.get(x.type, 10),
            x.line_number or 0
        ), reverse=True)
        
        return clustered_issues
    
    def _merge_with_escalation(self, issues: List[VerificationIssue]) -> VerificationIssue:
        """Merge issues with severity escalation logic"""
        # Use highest severity issue as primary
        primary_issue = max(issues, key=lambda x: self._severity_to_numeric(x.severity))
        
        # Collect agent sources
        agent_sources = set()
        for issue in issues:
            if hasattr(issue, 'agent_source'):
                agent_sources.add(issue.agent_source)
        
        # Escalate severity if multiple agents detect same issue
        escalated_severity = primary_issue.severity
        if len(agent_sources) > 1:
            # Multiple agents detecting same issue = escalate severity
            if escalated_severity == Severity.MEDIUM:
                escalated_severity = Severity.HIGH
            elif escalated_severity == Severity.LOW:
                escalated_severity = Severity.MEDIUM
        
        # Create enhanced message
        if len(agent_sources) > 1:
            message_suffix = f" (confirmed by {len(agent_sources)} agents: {', '.join(sorted(agent_sources))})"
            merged_message = primary_issue.message + message_suffix
        else:
            merged_message = primary_issue.message
        
        merged_issue = VerificationIssue(
            type=primary_issue.type,
            severity=escalated_severity,
            message=merged_message,
            line_number=primary_issue.line_number,
            column_number=primary_issue.column_number,
            file_path=primary_issue.file_path,
            suggestion=primary_issue.suggestion,
            confidence=min(1.0, primary_issue.confidence + 0.1 * len(agent_sources))
        )
        
        # Add metadata
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
    
    def _calculate_enterprise_score(self, agent_results: Dict[str, AgentResult], 
                                   issues: List[VerificationIssue]) -> float:
        """
        Calculate enterprise score with production deployment standards.
        
        Breakthrough features:
        - Security veto power
        - Compound vulnerability penalties
        - Production deployment blockers
        """
        if not agent_results:
            return 0.0
        
        # Step 1: Calculate base weighted score
        weighted_score = 0.0
        total_weight = 0.0
        
        for agent_name, result in agent_results.items():
            weight = self.agent_weights.get(agent_name, 0.25)
            
            if not result.success:
                agent_score = 0.0
            else:
                agent_score = result.overall_score
            
            weighted_score += agent_score * weight
            total_weight += weight
        
        base_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Step 2: Apply enterprise security veto power
        security_result = agent_results.get('security')
        if security_result and security_result.success and security_result.overall_score < 0.7:
            # Security veto: cap score at 0.4 if security fails enterprise standards
            base_score = min(base_score, 0.4)
        
        # Step 3: Apply production deployment penalties
        final_score = self._apply_production_penalties(base_score, issues, agent_results)
        
        return round(final_score, 3)
    
    def _apply_production_penalties(self, base_score: float, issues: List[VerificationIssue],
                                   agent_results: Dict[str, AgentResult]) -> float:
        """Apply enterprise production deployment penalties"""
        
        # Categorize issues by severity
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        high_issues = [i for i in issues if i.severity == Severity.HIGH]
        
        # Enterprise deployment blocker logic
        score = base_score
        
        # Critical issues: Each critical issue caps score significantly
        if critical_issues:
            # Multiple critical issues = exponential penalty
            critical_penalty = 0.6 + (len(critical_issues) - 1) * 0.2
            score = min(score, 1.0 - critical_penalty)
        
        # High issues: Production deployment risk
        if len(high_issues) >= 3:
            # 3+ high issues = production deployment blocker
            score = min(score, 0.4)
        elif len(high_issues) >= 2:
            # 2 high issues = significant concern
            score = min(score, 0.6)
        elif len(high_issues) >= 1:
            # 1 high issue = moderate concern  
            score = min(score, 0.8)
        
        # Compound vulnerability penalty
        compound_vulns = [i for i in issues if i.type in ['compound_vulnerability', 'security_cascade']]
        if compound_vulns:
            # Compound vulnerabilities are enterprise deployment blockers
            score = min(score, 0.3)
        
        # Failed agent penalty
        failed_agents = sum(1 for result in agent_results.values() if not result.success)
        if failed_agents > 0:
            failure_penalty = 0.15 * failed_agents  # 15% penalty per failed agent
            score = max(0.0, score - failure_penalty)
        
        return max(0.0, score)
    
    def _determine_production_status(self, overall_score: float, 
                                   issues: List[VerificationIssue]) -> str:
        """
        Determine production deployment status with enterprise standards.
        """
        # Enterprise deployment decision logic
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        high_issues = [i for i in issues if i.severity == Severity.HIGH]
        compound_vulns = [i for i in issues if i.type in ['compound_vulnerability', 'security_cascade']]
        
        # Immediate deployment blockers
        if critical_issues or compound_vulns:
            return "FAIL"
        
        # High-risk scenarios
        if len(high_issues) >= 3:
            return "FAIL"
        elif len(high_issues) >= 2:
            return "FAIL"  # More aggressive than before
        
        # Score-based decisions with enterprise thresholds
        enterprise_threshold = self.config.thresholds.get('overall_min_score', 0.85)
        
        if overall_score >= enterprise_threshold:
            # High score but check for remaining risks
            if len(high_issues) >= 1:
                return "WARNING"  # Even 1 high issue is concerning
            return "PASS"
        elif overall_score >= enterprise_threshold * 0.8:  # Within 20% of threshold
            return "WARNING"
        else:
            return "FAIL"
    
    def _generate_metadata(self, agent_results: Dict[str, AgentResult], 
                          issues: List[VerificationIssue], 
                          code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive metadata with enterprise metrics"""
        metadata = {
            'analysis_timestamp': time.time(),
            'code_stats': self._analyze_code_stats(code),
            'agent_summary': {},
            'issue_summary': self._summarize_issues(issues),
            'enterprise_metrics': self._calculate_enterprise_metrics(agent_results, issues),
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
        
        return metadata
    
    def _calculate_enterprise_metrics(self, agent_results: Dict[str, AgentResult],
                                    issues: List[VerificationIssue]) -> Dict[str, Any]:
        """Calculate enterprise-specific metrics"""
        metrics = {
            'production_readiness': 0.0,
            'security_posture': 0.0,
            'scalability_score': 0.0,
            'deployment_risk': 'LOW'
        }
        
        # Security posture
        security_result = agent_results.get('security')
        if security_result and security_result.success:
            security_issues = [i for i in issues if hasattr(i, 'agent_source') and i.agent_source == 'security']
            critical_security = [i for i in security_issues if i.severity == Severity.CRITICAL]
            
            if not critical_security:
                metrics['security_posture'] = security_result.overall_score
            else:
                metrics['security_posture'] = 0.2  # Critical security issues tank posture
        
        # Scalability score
        performance_result = agent_results.get('performance')
        if performance_result and performance_result.success:
            performance_issues = [i for i in issues if hasattr(i, 'agent_source') and i.agent_source == 'performance']
            complexity_issues = [i for i in performance_issues if 'complexity' in i.type]
            
            if not complexity_issues:
                metrics['scalability_score'] = performance_result.overall_score
            else:
                metrics['scalability_score'] = max(0.3, performance_result.overall_score - 0.4)
        
        # Deployment risk assessment
        critical_count = len([i for i in issues if i.severity == Severity.CRITICAL])
        high_count = len([i for i in issues if i.severity == Severity.HIGH])
        
        if critical_count > 0:
            metrics['deployment_risk'] = 'CRITICAL'
        elif high_count >= 2:
            metrics['deployment_risk'] = 'HIGH'
        elif high_count >= 1:
            metrics['deployment_risk'] = 'MEDIUM'
        else:
            metrics['deployment_risk'] = 'LOW'
        
        # Production readiness (overall)
        if metrics['deployment_risk'] == 'LOW' and metrics['security_posture'] > 0.8:
            metrics['production_readiness'] = 0.9
        elif metrics['deployment_risk'] == 'MEDIUM':
            metrics['production_readiness'] = 0.6
        elif metrics['deployment_risk'] == 'HIGH':
            metrics['production_readiness'] = 0.3
        else:
            metrics['production_readiness'] = 0.1
        
        return metrics
    
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
            'unique_types': len(type_counts),
            'compound_vulnerabilities': len([i for i in issues if 'compound' in i.type])
        }
    
    def _create_summary(self, agent_results: Dict[str, AgentResult], 
                       issues: List[VerificationIssue], 
                       overall_score: float) -> Dict[str, Any]:
        """Create comprehensive summary with enterprise insights"""
        
        # Agent performance summary
        agent_summary = {}
        for agent_name, result in agent_results.items():
            # More stringent pass criteria
            agent_threshold = 0.85 if agent_name == 'security' else 0.80
            status = "✓ PASS" if result.success and result.overall_score >= agent_threshold else "✗ FAIL"
            agent_summary[agent_name] = {
                'status': status,
                'score': result.overall_score,
                'issues_found': len(result.issues),
                'execution_time': f"{result.execution_time:.3f}s"
            }
        
        # Enterprise issue breakdown
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        high_issues = [i for i in issues if i.severity == Severity.HIGH]
        compound_issues = [i for i in issues if 'compound' in i.type]
        
        # Enterprise key findings
        key_findings = []
        if compound_issues:
            key_findings.append(f"{len(compound_issues)} compound vulnerabilities detected")
        if critical_issues:
            key_findings.append(f"{len(critical_issues)} critical deployment blockers")
        if high_issues:
            key_findings.append(f"{len(high_issues)} high-priority production risks")
        if len(issues) == 0:
            key_findings.append("Enterprise-ready: No deployment blockers detected")
        
        return {
            'overall_score': overall_score,
            'grade': self._score_to_enterprise_grade(overall_score),
            'agent_performance': agent_summary,
            'issue_breakdown': {
                'critical': len(critical_issues),
                'high': len(high_issues),
                'medium': len([i for i in issues if i.severity == Severity.MEDIUM]),
                'low': len([i for i in issues if i.severity == Severity.LOW]),
                'compound': len(compound_issues),
                'total': len(issues)
            },
            'key_findings': key_findings,
            'enterprise_assessment': self._assess_enterprise_readiness(issues, overall_score),
            'top_issue_types': self._get_top_issue_types(issues)
        }
    
    def _score_to_enterprise_grade(self, score: float) -> str:
        """Convert numeric score to enterprise grade"""
        if score >= 0.95:
            return "A+ (Enterprise Ready)"
        elif score >= 0.90:
            return "A (Production Ready)"
        elif score >= 0.85:
            return "A- (Minor Concerns)"
        elif score >= 0.75:
            return "B+ (Needs Review)"
        elif score >= 0.65:
            return "B (Multiple Issues)"
        elif score >= 0.50:
            return "C (Major Rework)"
        elif score >= 0.30:
            return "D (Deployment Blocked)"
        else:
            return "F (Critical Failures)"
    
    def _assess_enterprise_readiness(self, issues: List[VerificationIssue], score: float) -> Dict[str, Any]:
        """Assess enterprise deployment readiness"""
        critical_count = len([i for i in issues if i.severity == Severity.CRITICAL])
        high_count = len([i for i in issues if i.severity == Severity.HIGH])
        security_issues = len([i for i in issues if hasattr(i, 'agent_source') and i.agent_source == 'security'])
        
        if critical_count == 0 and high_count == 0 and score >= 0.9:
            readiness = "READY"
            confidence = "HIGH"
            blockers = []
        elif critical_count == 0 and high_count <= 1 and score >= 0.8:
            readiness = "CONDITIONAL"
            confidence = "MEDIUM" 
            blockers = ["Minor issues need resolution"]
        elif critical_count > 0:
            readiness = "BLOCKED"
            confidence = "HIGH"
            blockers = [f"{critical_count} critical issues must be resolved"]
        else:
            readiness = "BLOCKED"
            confidence = "HIGH"
            blockers = ["Multiple high-priority issues"]
        
        return {
            'deployment_readiness': readiness,
            'confidence': confidence,
            'blocking_issues': blockers,
            'security_risk_level': 'HIGH' if security_issues > 2 else 'MEDIUM' if security_issues > 0 else 'LOW'
        }
    
    def _get_top_issue_types(self, issues: List[VerificationIssue], top_n: int = 5) -> List[Dict[str, Any]]:
        """Get top N most critical issue types"""
        if not issues:
            return []
        
        type_counts = Counter(issue.type for issue in issues)
        top_types = []
        
        for issue_type, count in type_counts.most_common(top_n):
            type_issues = [i for i in issues if i.type == issue_type]
            worst_severity = max(type_issues, key=lambda x: self._severity_to_numeric(x.severity)).severity
            avg_confidence = sum(getattr(i, 'confidence', 1.0) for i in type_issues) / len(type_issues)
            
            top_types.append({
                'type': issue_type,
                'count': count,
                'worst_severity': worst_severity.value,
                'avg_confidence': round(avg_confidence, 2)
            })
        
        return top_types
    
    def _generate_recommendations(self, issues: List[VerificationIssue], 
                                agent_results: Dict[str, AgentResult]) -> List[str]:
        """Generate enterprise-focused recommendations"""
        recommendations = []
        
        # Critical enterprise blockers first
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        compound_issues = [i for i in issues if 'compound' in i.type]
        
        if critical_issues or compound_issues:
            recommendations.append("🚨 DEPLOYMENT BLOCKED: Resolve critical security/correctness issues immediately")
            for issue in (critical_issues + compound_issues)[:3]:
                if issue.suggestion:
                    recommendations.append(f"   • {issue.suggestion}")
        
        # Security enterprise recommendations
        security_issues = [i for i in issues if hasattr(i, 'agent_source') and i.agent_source == 'security']
        if len(security_issues) >= 2:
            recommendations.append("🔒 SECURITY PRIORITY: Multiple security vulnerabilities detected")
            recommendations.append("   • Conduct comprehensive security review before deployment")
            recommendations.append("   • Implement security testing in CI/CD pipeline")
        
        # Performance scalability recommendations
        perf_result = agent_results.get('performance')
        if perf_result and perf_result.overall_score < 0.6:
            recommendations.append("⚡ SCALABILITY RISK: Performance issues will impact production scale")
            recommendations.append("   • Conduct load testing with realistic data volumes")
            recommendations.append("   • Optimize algorithms before deployment")
        
        # Enterprise success recommendations
        if not critical_issues and len([i for i in issues if i.severity == Severity.HIGH]) == 0:
            recommendations.append("✅ ENTERPRISE READY: Code meets production deployment standards")
            recommendations.append("💡 Next: Implement monitoring and alerting for production")
        
        return recommendations[:8]  # Limit to top 8 most critical